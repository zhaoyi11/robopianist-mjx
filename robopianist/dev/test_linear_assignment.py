from absl.testing import absltest
import numpy as np
import scipy
import scipy.optimize
import jax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import jit
from jax import config
from linear_sum_assignment import linear_sum_assignment
config.parse_flags_with_absl()
def rosenbrock(np):
  def func(x):
    return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)
  return func
def himmelblau(np):
  def func(p):
    x, y = p
    return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2
  return func
def matyas(np):
  def func(p):
    x, y = p
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
  return func
def eggholder(np):
  def func(p):
    x, y = p
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
      np.sqrt(np.abs(x - (y + 47.))))
  return func
def zakharovFromIndices(x, ii):
  sum1 = (x**2).sum()
  sum2 = (0.5*ii*x).sum()
  answer = sum1+sum2**2+sum2**4
  return answer

class TestLSA(jtu.JaxTestCase):
  """
  Tests for linear_sum_assignment.
  """

  @jtu.sample_product(
    maximize=[False, True],
    shape=[(10, 10), (100, 60), (60, 100)],
    dtype=["float32", "int32"],
  )
  def test_lsa(self, maximize, shape, dtype):

    costs = jtu.rand_default(self.rng())([10, *shape], dtype)

    scipy_res = [scipy.optimize.linear_sum_assignment(
        cost,
        maximize=maximize
      ) for cost in costs]
    jax_res = [linear_sum_assignment(
        cost,
        maximize=maximize
      ) for cost in costs]

    self.assertAllClose(jax_res, scipy_res)

  @jtu.sample_product(
    maximize=[False, True],
    shape=[(10, 10), (100, 60), (60, 100)],
    dtype=["float32", "int32"],
  )
  def test_transform(self, maximize, shape, dtype):

    costs = jtu.rand_default(self.rng())([10, *shape], dtype)

    scipy_res = [scipy.optimize.linear_sum_assignment(
        cost,
        maximize=maximize
      ) for cost in costs]
    lsa_ = linear_sum_assignment
    lsa = lambda cost : lsa_(cost, maximize=maximize)
    lsa = jax.vmap(lsa)
    lsa = jax.jit(lsa)
    jax_res = lsa(costs)
    self.assertAllClose(
      jnp.array(jax_res).transpose(1,0,2),
      jnp.array(scipy_res)
    )

if __name__ == "__main__":
  absltest.main()