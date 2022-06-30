"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch as tc
from torch import nn, optim


# If you want to use double precision for the computations (torch uses single precision
# by default):
# tc.set_default_tensor_type(tc.DoubleTensor)


def blk_tridag_chol(A, B, upper=True):
    """
    Compute the cholesky decompoisition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (lower/upper) 1st block
        off-diagonal matrix. (Originally it only accepted the upper 1st block off-
        diagonal matrix)

    Outputs:
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky

    """
    T = A.shape[0]
    n = A.shape[1]
    assert (A.shape == (T, n, n))
    assert (B.shape == (T - 1, n, n))

    # Code for computing the cholesky decomposition of a symmetric block tridiagonal matrix
    def compute_chol(Aip1, Bi, Li, upper: bool):
        if upper:
            Ci = Bi.t() @ tc.inverse(Li).t()
        else:
            Ci = Bi @ tc.inverse(Li).t()

        Dii = Aip1 - Ci @ Ci.t()
        Lii = tc.cholesky(Dii)
        return [Lii, Ci]

    L = tc.empty_like(A)
    C = tc.empty_like(B)

    # By default, torch.cholesky returns lowerdiagonal matrix L, such that X = L @ L.t()
    # The same holds for the theano implementation of cholesky.
    try:
        L[0] = tc.cholesky(A[0])
    except:
        print("Error in computing torch.cholesky, most likely singular U.")
        print(A)
        print(B)

    C[0] = tc.zeros_like(B[0])
    for t in range(0, len(B)):  # T-1
        L[t + 1], C[t] = compute_chol(A[t + 1], B[t], L[t], upper)

    return [L, C]


def blk_chol_inv(A, B, b, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)

    Outputs:
    x - solution of Cx = b
    
    ###############################################################################################

    Edit: 
    A.shape=(T, dim_z, dim_z), B.shape=(T-1, dim_z, dim_z), b.shape=(T, dim_z), output.shape=(T, dim_z)

    The helper function 'blk_chol_inv(A, B, b, lower=True, transpose=False)' solves the equation 
    Cx = b for x, where C is a block bidiagonal matrix with diagonal blocks A and offdiagonal blocks B. 
    If transpose=True, then it solves the equation C.t() x = b. The flag 'lower' tells the function
    how to interprete the offdiagonal elements B, i.e., if they should be treated as upper or lower
    diagonal blocks.
    In our case we have the equation 'z = mu + (R^-1).t() epsilon', where R is a lowerdiagonl cholesky
    factor and therefore a lower bidiagonal block matrix (where the diagonal blocks are lower diagonal 
    as well). We are therefore interested in solving the equation R.t() x = epsilon, i.e., we want to
    calculate x = (R^-1).t() epsilon. Thus, we can identify C.t()=R.t() and b=epsilon. Since we have a 
    transposed matrix, we need to set the flag 'transpose=True' as we want to solve C.t() x = b.
    Also, we know that R is a lower bidiagonal matrix and hence R.t() is an upper bidiagonal matrix. 
    Hence, the helper function must treat R.t() as an upper bidiagonal matrix. 
    We could also transpose R before calling the function and then handing R.t() over to the function.
    Then, we would need to set 'upper=True' as it is an upperdiagonal matrix but 'transpose=False', 
    since the matrix already has been transposed outside of the function. This is because in the function,
    the offdiagonal blocks are first transposed (if the flag is set to True) and then afterwards the function 
    just treats these blocks as a lower or upper diagonal depending on the flag 'lower'. I.e., these two
    paramters 'lower' and 'transpose' are not really connected to each other.
    """

    T = A.shape[0]
    n = A.shape[1]

    if transpose:
        A = tc.einsum('tjk->tkj', A)
        B = tc.einsum('tjk->tkj', B)
    if lower:
        x = tc.zeros((T, n))
        bt = tc.zeros((T, n))
        x[0] = tc.inverse(A[0]) @ b[0]

        def lower_step(Akp1, Bk, bkp1, xk):
            return tc.inverse(Akp1) @ (bkp1 - Bk @ xk)

        for t in range(T - 1):
            x[t + 1] = lower_step(A[t + 1], B[t], b[t + 1], x[t].clone())  # clone prevents inplace error
    else:
        x = tc.zeros((T, n))
        x[-1] = tc.inverse(A[-1]) @ b[-1]

        def upper_step(Akm1, Bkm1, bkm1, xk):
            return tc.inverse(Akm1) @ (bkm1 - Bkm1 @ xk)

        for t in range(T - 1):
            x[-t - 2] = upper_step(A[-t - 2], B[-t - 1], b[-t - 2], x[-t - 1].clone())
    return x


def blk_chol_mtimes(A, B, x, lower=True, transpose=False):
    """
    Evaluate Cx = b, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)

    Outputs:
    b - result of Cx = b

    """
    print('This function is not tested')
    raise NotImplementedError
    T = A.shape[0]
    n = A.shape[1]
    assert (A.shape == (T, n, n))
    assert (B.shape == (T - 1, n, n))
    assert (x.shape == (T, n))
    if transpose:
        A = A.dimshuffle(0, 2, 1)
        B = B.dimshuffle(0, 2, 1)
    if lower:
        b = tc.empty((T, n))
        b[0] = A[0] @ x[0]

        def lower_step(Ak, Bkm1, xkm1, xk):
            return Bkm1 @ xkm1 + Ak @ xk

        for t in range(T - 1):
            b[t + 1] = lower_step(A[t + 1], B[t], x[t], x[t + 1])
    else:
        raise NotImplementedError

    return b


def calculate_cholesky_factors(mu, cov, dim_z, QinvChol, Q0invChol, A, batch_size):
    # self.LambdaChol has shape (batch_size, dim_z, dim_z)
    LambdaChol = cov.view(-1, dim_z, dim_z)
    # self.Lambda has shape (batch_size, dim_z, dim_z)
    Lambda = tc.bmm(LambdaChol, LambdaChol.permute(0, 2, 1))
    # self.Qinv and self.Q0inv have shape (dim_z, dim_z)
    Qinv = tc.matmul(QinvChol, QinvChol.t())
    Q0inv = tc.matmul(Q0invChol, Q0invChol.t())

    # AQinvA has shape (dim_z, dim_z)
    AQinvA = tc.matmul(tc.matmul(A.t(), Qinv), A)
    AQinvrep = -tc.matmul(A.t(), Qinv)

    # 'torch.repeat' stacks the tensor 'AQinvrep' (batch_size-1) times on top
    # of each other, resulting in a (batch_size-1, dim_z, dim_z) tensor
    # where each AQinvrep[i] contains the (dim_z, dim_z) tensor 'AQinvrep'
    # as calculated above
    AQinvrep = AQinvrep.repeat(batch_size - 1, 1, 1)
    AQinvArep = AQinvA + Qinv
    AQinvArep = AQinvArep.repeat(batch_size - 2, 1, 1)

    # AQinvArepPlusQ has shape (batch_dim, dim_z, dim_z)
    AQinvArepPlusQ = tc.cat(((Q0inv + AQinvA).unsqueeze(0), AQinvArep, Qinv.
                             unsqueeze(0)))
    # AA are the diagonal blocks of the covariance matrix and BB are the off-diagonal blocks.
    AA = Lambda + AQinvArepPlusQ
    BB = AQinvrep

    # self.Lambda has shape (batch_size, dim_z, dim_z), self.Mu has shape (batch_size, dim_z).
    # To use torch.bmm, self.Mu also has to be a 3D tensor, hence we have to add a third
    # dimension. Then self.Mu.unsqueeze(2) will have shape (batch_size, dim_z, 1).
    # LambdaMu has shape (batch_size, dim_z, 1)
    LambdaMu = tc.bmm(Lambda, mu.unsqueeze(2))

    # After calculating 'bmm', we have to get rid of the additional dimension and reshape
    # LambdaMu from (batch_size, dim_z, 1) to (batch_size, dim_z)
    LambdaMu = LambdaMu.view(-1, dim_z)

    return AA, BB, LambdaMu


def weights_init_uniform_rule(m: nn.Module) -> None:
    """takes in a module and applies the specified weight initialization

    Args:
     m: Model instance
    """
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    # TODO: right now isinstance is always false, therefore weight init is not
    #  working at all. Only done in the init of encoder/decoder models.
    if isinstance(classname, nn.Linear):
        print("Applying uniform weights")
        y = 0.0008
        m.weight.data.uniform_(1 - y, 1 + y)
        m.bias.data.fill_(0)
    return


def addGaussianNoiseToData(data, percentage):
    noise = tc.randn(data.shape)
    data = data * (1. - percentage) + data * noise * percentage
    return data


if __name__ == "__main__":
    print('oh yeah....')

    # Build a block tridiagonal matrix
    npA = np.mat('1  .9; .9 4', dtype=float)
    npB = .01 * np.mat('2  7; 7 4', dtype=float)
    npC = np.mat('3  0; 0 1', dtype=float)
    npD = .01 * np.mat('7  2; 9 3', dtype=float)
    npE = .01 * np.mat('2  0; 4 3', dtype=float)
    npF = .01 * np.mat('1  0; 2 7', dtype=float)
    npG = .01 * np.mat('3  0; 8 1', dtype=float)

    npZ = np.mat('0 0; 0 0')

    lowermat = np.bmat([[npF, npZ, npZ, npZ],
                        [npB.T, npC, npZ, npZ],
                        [npZ, npD.T, npE, npZ],
                        [npZ, npZ, npB.T, npG]])
    print(lowermat)
    # tlower = tc.tensor(lowermat)

    # numpy uses double precision (float64) by default whereas torch uses single precision
    # (float32) by default. Therefore, when transforming numpy arrays to torch tensors, 
    # these torch tensors have float64 as dtype. Hence one has to cast them to float32
    # since otherwise there will be errors when using e.g. 'matmul' with an argument of
    # dtype float32 and one of dtype float64
    tA = tc.tensor(npA).float()
    tB = tc.tensor(npB).float()
    tC = tc.tensor(npC).float()
    tD = tc.tensor(npD).float()
    tE = tc.tensor(npE).float()
    tF = tc.tensor(npF).float()
    tG = tc.tensor(npG).float()
    print(tA.dtype)
    print(tB.dtype)

    theD = tc.stack(tensors=(tF, tC, tE, tG), dim=0)
    theOD = tc.stack(tensors=(tB.t(), tD.t(), tB.t()), dim=0)

    npb = np.mat('1 2; 3 4; 5 6; 7 8', dtype=float)
    print(npb)
    tb = tc.tensor(npb).float()

    cholmat = lowermat.dot(lowermat.T)
    print(cholmat.shape)
    # cholmat = lower @ tlower.t()

    print('tb {}'.format(tb.shape))
    print('tb = ', tb.shape)
    print('theD = ', theD.shape)
    print('theOD = ', theOD.shape)

    ib = blk_chol_inv(theD, theOD, tb)
    print('ib.shape = ', ib.shape)

    npb2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    print(np.linalg.inv(lowermat).dot(npb2))
    print('theD = ', theD.shape)
    print('theOD = ', theOD.shape)
    print('ib = ', ib.shape)
    x = blk_chol_inv(theD, theOD, ib, lower=False, transpose=True)
    print('x.shape = ', x.shape)
    print(np.linalg.inv(lowermat.T).dot(ib.flatten()))
    print('x shape: {}'.format(x.shape))
    print('x {}'.format(x.flatten()))
    print(x.flatten())
    print(np.linalg.inv(cholmat).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8])))
    print('Cholesky inverse matches numpy inverse: ',
          np.allclose(x.flatten(), np.linalg.inv(cholmat).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]))))
