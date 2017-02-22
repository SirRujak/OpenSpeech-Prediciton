'''
Copyright (c) 2016, Daniel McNeela
All rights reserved. 

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Retina nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.metrics.pairwise import euclidean_distances

def sammon(data, target_dim=2, max_iterations=250, max_halves=10):
    """
    Adopted from the Matlab implementation by Dr. Gavin C. Cawley.
    Matlab source can be found here:

    https://people.sc.fsu.edu/~jburkardt/m_src/profile/sammon_test.m
    """
    TolFun = 1 * 10 ** (-9)

    D = euclidean_distances(data, data)
    N = data.shape[0]
    scale = np.sum(D.flatten('F'))
    D = D + np.identity(N)
    D_inv = np.linalg.inv(D)

    y = np.random.randn(N, target_dim)
    one = np.ones((N, target_dim))
    d = euclidean_distances(y, y) + np.identity(N)
    d_inv = np.linalg.inv(d)
    delta = D - d
    E = np.sum(np.sum(np.power(delta, 2) * D_inv))

    for i in range(max_iterations):
        delta = d_inv - D_inv
        deltaone = np.dot(delta, one)
        g = np.dot(delta, y) - y * deltaone
        dinv3 = np.power(d_inv, 3)
        y2 = np.power(y, 2)
        H = np.dot(dinv3, y2) - deltaone - 2 * np.multiply(y, np.dot(dinv3, y)) + np.multiply(y2, np.dot(dinv3, one))
        s = np.divide(-np.transpose(g.flatten('F')), np.transpose(np.abs(H.flatten('F'))))
        y_old = y

    for j in range(max_halves):
        [rows, columns] = y.shape
        y = y_old.flatten('F') + s
        y = y.reshape(rows, columns)
        d = euclidean_distances(y, y) + np.identity(N)
        d_inv = np.linalg.inv(d)
        delta = D - d
        E_new = np.sum(np.sum(np.power(delta, 2) * D_inv))

        if E_new < E:
            break
        else:
            s = 0.5 * s

    E = E_new
    E = E * scale
    return (y, E)
