from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

def _var_key(var):
  # TODO(ashankar): Consolidate handling for eager and graph
  if hasattr(var, "op"):
    return (var.op.graph, var.op.name)
  return var._unique_id  # pylint: disable=protected-access

class AcclipOptimizer(optimizer.Optimizer):
    def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.99,
               alpha=2,
               epsilon=1e-8,
               use_locking=False,
               name="Acclip"):
        self.s=super(AcclipOptimizer, self)
        self.s.__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._alpha = alpha
        self._epsilon = epsilon
        self._learning_rate_tensor = None
        
    def compute_gradients(self, loss, var_list=None, **kwargs):
         
        return self.s.compute_gradients(loss, var_list)
     
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        with ops.init_scope():
            self._create_slots1([v for g, v in grads_and_vars])
        index=0
        grads_and_vars=list(grads_and_vars)
        for g, var in grads_and_vars:
            print('value',var)
            clip=self.s.get_slot(var,'clip')
            m=self.s.get_slot(var,'m')
            v=self.s.get_slot(var,'v')
            m = m*self._beta1+(1-self._beta1)*g
            clip=clip*self._beta2+(1-self._beta2)*(tf.pow(tf.math.abs(g),self._alpha))
            v=v*self._beta2+(1-self._beta2)*g*g 
            denom=tf.clip_by_value((tf.pow(clip,1/self._alpha))/(tf.math.abs(m)+self._epsilon),0.0,1.0)
            denom=tf.math.sqrt(denom/((v*self._beta2))+self._epsilon)
            g=g*denom*m
            grads_and_vars[index]=(g,var)
            index=index+1
            self._update_slot('clip', var, clip)
            self._update_slot('m', var, m)
            self._update_slot('v', var, v)
            
        return self.s.apply_gradients(grads_and_vars, global_step, name)
    
    def compute_gradients_adam(self, loss, global_step, var_list=None, **kwargs):
        grads_and_vars = self.s.compute_gradients(loss, var_list)
        with ops.init_scope():
            self._create_slots1([v for g, v in grads_and_vars])
        index=0
        grads_and_vars=list(grads_and_vars)
        t = global_step +1
        for g, var in grads_and_vars:
            print('var={}'.format(var),"name=",var.name)
            m=self.s.get_slot(var,'m')
            v=self.s.get_slot(var,'v')
             
             
#             alpha = self._lr*np.sqrt(1 - self._beta2)/(1 - self._beta1)
            alpha = tf.math.sqrt((1 - tf.pow(self._beta2, t))/(1 - tf.pow(self._beta1, t)))
            m = m + (g-m)*(1-self._beta1)
            v = v+ (tf.math.square(g) - v) * (1 - self._beta2)
            m1 = m/(1-tf.pow(self._beta1,global_step+1))
            v1 = v/(1 - tf.pow(self._beta2,global_step+1))
            if False:
                g = ((g * (1 - self._beta1) + self._beta1 * m) * alpha) / (tf.math.sqrt(v) + self._epsilon);
            else:   
                g = (m1 * alpha) / (tf.math.sqrt(v1) + self._epsilon)
            grads_and_vars[index]=(g,var)
             
            index=index+1
            self._update_slot('m', var, m)
            self._update_slot('v', var, v)
        return grads_and_vars
    
    def apply_gradients_adam(self, grads_and_vars, global_step=None, name=None):
#         with ops.init_scope():
#             self._create_slots1([v for g, v in grads_and_vars])
#         index=0
#         grads_and_vars=list(grads_and_vars)
#         t = global_step+1
#         
#         for g, var in grads_and_vars:
#             print('value',var)
#             m=self.s.get_slot(var,'m')
#             v=self.s.get_slot(var,'v')
#             
#             
# #             alpha = self._lr*np.sqrt(1 - self._beta2)/(1 - self._beta1)
#             alpha = tf.math.sqrt((1 - tf.pow(self._beta2, t))/(1 - tf.pow(self._beta1, t)))
#             m = m + (g-m)*(1-self._beta1)
#             v = v+ (tf.math.square(g) - v) * (1 - self._beta2)
#             m1 = m/(1-tf.pow(self._beta1,global_step+1))
#             v1 = v/(1 - tf.pow(self._beta2,global_step+1))
#             if False:
#                 g = ((g * (1 - self._beta1) + self._beta1 * m) * alpha) / (tf.math.sqrt(v) + self._epsilon);
#             else:   
#                 g = (m1 * alpha) / (tf.math.sqrt(v1) + self._epsilon)
#             grads_and_vars[index]=(g,var)
#             
#             index=index+1
#             self._update_slot('m', var, m)
#             self._update_slot('v', var, v)
            
        return self.s.apply_gradients(grads_and_vars, global_step, name)
    
    def apply_gradients2(self, grads_and_vars, global_step=None, name=None):
        with ops.init_scope():
            self._create_slots1([v for g, v in grads_and_vars])
        index=0
        grads_and_vars=list(grads_and_vars)
        m_and_vs=[]
        for g, var in grads_and_vars:
            print('value',var)
            clip=self.s.get_slot(var,'clip')
            m=self.s.get_slot(var,'m')
            v=self.s.get_slot(var,'v')
            m = m*self._beta1+(1-self._beta1)*g
            clip=clip*self._beta2+(1-self._beta2)*(tf.pow(tf.math.abs(g),self._alpha))
            v=v*self._beta2+(1-self._beta2)*g*g 
            denom=tf.clip_by_value((tf.pow(clip,1/self._alpha))/(tf.math.abs(m)+self._epsilon),0.0,1.0)
            denom=tf.math.sqrt(denom/((v*self._beta2))+self._epsilon)
            g=g*denom*m
            grads_and_vars[index]=(g,var)
            index=index+1
            m_and_vs.append((m,v))
#             self._update_slot('clip', var, clip)
#             self._update_slot('m', var, m)
#             self._update_slot('v', var, v)
            
        return self.s.apply_gradients(grads_and_vars, global_step, name),m_and_vs
    
    
    def _create_slots1(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
#         first_var = min(var_list, key=lambda x: x.name)
#         self._create_non_slot_variable(
#             initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
#         self._create_non_slot_variable(
#             initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
     
        # Create slots for the first and second moments.
        for v in var_list:
          self._zeros_slot(v, "m", self._name)
          self._zeros_slot(v, "v", self._name)
          self._zeros_slot(v, "clip", self._name)
    
    def _update_slot(self,name,var,value):
        named_slots = self.s._slot_dict(name)
        named_slots[_var_key(var)]=value
#         named_slots[var]=value
    
    def _apply_dense(self, grad, var):
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op
    def _prepare(self):
        learning_rate = self._call_if_callable(self._lr)
        self._learning_rate_tensor = ops.convert_to_tensor(
            learning_rate, name="learning_rate")
#     def _resource_apply_dense(self, grad, handle):
#         return training_ops.resource_apply_gradient_descent(
#             handle.handle, math_ops.cast(self._learning_rate_tensor,
#                                          grad.dtype.base_dtype),
#             grad, use_locking=self._use_locking)
# 
#     def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
#         return resource_variable_ops.resource_scatter_add(
#             handle.handle, indices, -grad * self._learning_rate)
# 
#     def _apply_sparse_duplicate_indices(self, grad, var):
#         delta = ops.IndexedSlices(
#             grad.values *
#             math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
#             grad.indices, grad.dense_shape)
#         return var.scatter_sub(delta, use_locking=self._use_locking)
    
