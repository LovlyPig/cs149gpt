#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <cmath>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] = val;
}

/*
一个形状为 `[N, C, H, W]`（分别对应批次、通道、高度、宽度）的 4D 张量在内存中实际上存储为**连续的一维数据块**。从 4D 坐标到线性地址的映射遵循固定的**步长规则**，其中变化最快的维度（最内层）步长为 1。主流布局有两种：

- **NCHW（通道优先）**：维度顺序为 `(N, C, H, W)`。`W` 变化最快，其次是 `H`，然后是 `C`，最后是 `N`。偏移量计算公式：  
  `((n × C + c) × H + h) × W + w`

- **NHWC（通道最后）**：维度顺序为 `(N, H, W, C)`。`C` 变化最快。偏移量计算公式：  
  `((n × H + h) × W + w) × C + c`

### 为什么选择这些布局？如何利用硬件特性？

**1. 空间局部性与缓存效率**  
在 CNN 中，卷积或池化等操作通常沿空间维度（`H`、`W`）滑动窗口。当 `W` 是变化最快的维度时（如 NCHW），同一行内相邻的像素在内存中彼此紧邻，便于**硬件预取**并充分利用**缓存行**。而在 NHWC 中，同一像素的所有通道值紧密排布，有利于**逐通道操作**或**逐像素的向量化计算**，减少跨通道访问时的内存跳跃。

**2. SIMD / 向量化**  
现代 CPU 和 GPU 擅长用单条指令加载连续的 4、8 或 16 个浮点数。  
- **NCHW** 倾向于**空间维度的向量化**（一次处理一行中的多个像素）。  
- **NHWC** 倾向于**通道维度的向量化**（用宽 SIMD 寄存器一次处理一个像素的所有通道）。  
这也是 cuDNN、TensorRT 等 GPU 库对特定卷积操作优先选用 NHWC 的原因之一。

**3. GPU 上的合并内存访问**  
GPU 以较大的块（如 32 或 128 字节）发起内存事务。在 NHWC 布局中，不同线程若访问相同 `(h, w)` 位置但不同通道的数据，这些数据在内存中是**连续**的，从而最大化**带宽利用率**（合并访问）。而 NCHW 布局下若不加细致的分块处理，容易出现跨步、非合并的读取，降低效率。

**4. 硬件加速器对齐**  
专用的 AI 硬件（TPU、NPU）往往强制要求特定的数据布局，以匹配其脉动阵列或张量核心的数据流。例如，TPU 输入要求采用 NHWC，这样通道数据可以直接流入矩阵乘法单元，无需额外的转置开销。

**总结**  
4D 张量的内存布局是一种刻意设计的权衡：通过让存储顺序与最常见操作的访问模式对齐，从而**减少内存延迟、饱和内存带宽、高效填充并行执行单元**。

*/ 

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten(); // 展平为 1D 张量
    tensor = tensor.contiguous(); // 确保内存连续
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */


// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

void matrixMutiplyTranspose(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int b, int h, int H, int N, int d) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < d; k++) {
                float valA = fourDimRead(A, b, h, i, k, H, N, d);
                float valB = fourDimRead(B, b, h, j, k, H, N, d);
                sum += valA * valB;
            }
            twoDimWrite(C, i, j, N, sum);
        }
    }
}

void matrixMutiply(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int b, int h, int H, int N, int d) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                float valA = twoDimRead(A, i, k, N);
                float valB = fourDimRead(B, b, h, k, j, H, N, d);
                sum += valA * valB;
            }
            fourDimWrite(C, b, h, i, j, H, N, d, sum);
        }
    }
}

void softmax(std::vector<float> &a, int N) {
    std::vector<float> maxval(N, 0.0);
    for (int i = 0; i < N; i++) {
        float maxVal = 0.0;
        for (int j = 0; j < N; j++) {
            float val = twoDimRead(a, i, j, N);
            if (val > maxVal) {
                maxVal = val;
            }
        }
        maxval[i] = maxVal;
    }


    for (int i = 0; i < N; i++) {
        float sum = 0.0;
        for (int j = 0; j < N; j++) {
            float val = twoDimRead(a, i, j, N);
            val = exp(val - maxval[i]);
            sum += val;
        }

        for (int j = 0; j < N; j++) {
            float val = twoDimRead(a, i, j, N);
            val = exp(val - maxval[i]) / sum;
            twoDimWrite(a, i, j, N, val);
        }
    }
}

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            matrixMutiplyTranspose(Q, K, QK_t, i, j, H, N, d);
            softmax(QK_t, N);
            matrixMutiply(QK_t, V, O, i, j, H, N, d);
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

// cache line size bytes
#define BLOCK_SIZE 64
// part1 115.908
// test 2 3 5 6 7 8 1 10
// 162.731 148.398 138.5 127.339 128.094 127.946 189.294
#define STRIDE 10

void blockedMatrixMultiplyTranspose(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int b, int h, int H, int N, int d) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < d; k++) {
                float valA = fourDimRead(A, b, h, i, k, H, N, d);
                float valB = fourDimRead(B, b, h, j, k, H, N, d);
                sum += valA * valB;
            }
            twoDimWrite(C, i, j, N, sum);
        }
    }

}

void blockedMatrixMultiply(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int b, int h, int H, int N, int d) {
    int tileSize = BLOCK_SIZE / sizeof(float);
    
    // for (int i = 0; i < N; i += STRIDE) {
    //     for (int t = 0; t < N; t += tileSize) {
    //         for (int j = 0; j < d; j++) {
    //             for (int ii = i; ii < std::min(i+STRIDE, N); ii++) {
    //                 float sum = 0.0;
    //                 for (int k = 0; k < tileSize; k++) {
    //                     int idx = t + k;
    //                     if (idx >= N) break;
    //                     float valA = twoDimRead(A, ii, idx, N);
    //                     float valB = fourDimRead(B, b, h, idx, j, H, N, d);
    //                     sum += valA * valB;
    //                 }
    //                 sum += fourDimRead(C, b, h, ii, j, H, N, d);
    //                 fourDimWrite(C, b, h, ii, j, H, N, d, sum);
    //             }
    //         }
    //     }
    // }

    // A B C 三个矩阵都是按照cache line 访问
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += tileSize) {
            for (int k = 0; k < d; k += tileSize) {
                for (int jj = j; jj < std::min(j + tileSize, N); jj++) {
                    for (int kk = k; kk < std::min(k + tileSize, d); kk++) {
                        float valA = twoDimRead(A, i, jj, N);
                        float valB = fourDimRead(B, b, h, jj, kk, H, N, d);
                        float valC = fourDimRead(C, b, h, i, kk, H, N, d);
                        valC += valA * valB;
                        fourDimWrite(C, b, h, i, kk, H, N, d, valC);
                    }
                }
            }

        }
    }
}

template<typename T>
void checkQKT(std::vector<T>& QK_t1, std::vector<T>& QK_t2, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T val1 = twoDimRead(QK_t1, i, j, N);
            T val2 = twoDimRead(QK_t2, i, j, N);
            if (std::abs(val1 - val2) > 1e-4) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " << val1 << " vs " << val2 << std::endl;
                exit(1);
            }
        }
    }
}

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            QK_t.assign(N * N, 0.0); // Reset QK_t for each head
            
            blockedMatrixMultiplyTranspose(Q, K, QK_t, i, j, H, N, d);
            softmax(QK_t, N);
            blockedMatrixMultiply(QK_t, V, O, i, j, H, N, d);
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);

                float maxval = -std::numeric_limits<float>::infinity();

                for (int j = 0; j < N; j++) {
                    float sum = 0.0;
                    for (int k = 0; k < d; k++) {
                        float valQ = fourDimRead(Q, b, h, i, k, H, N, d);
                        float valK = fourDimRead(K, b, h, j, k, H, N, d);
                        sum += valQ * valK;
                    }
                    ORow[j] = sum;
                    maxval = std::max(maxval, sum);
                }

                float sumExp = 0.0;
                for (int j = 0; j < N; j++) {
                    ORow[j] = exp(ORow[j] - maxval);
                    sumExp += ORow[j];
                }

                #pragma omp parallel for
                for (int j = 0; j < N; j++) {
                    ORow[j] /= sumExp;
                }

                for (int k = 0; k < d; k++) {
                    float sum = 0.0;
                    for (int j = 0; j < N; j++) {
                        float valORow = ORow[j];
                        float valV = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += valORow * valV;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                 }

		//YOUR CODE HERE
            }
	    }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // 设置每个线程的局部变量，避免false sharing
            std::vector<float> KjLocal = Kj;
            std::vector<float> VjLocal = Vj;
            std::vector<float> QiLocal = Qi;
            std::vector<float> OiLocal = Oi;
            std::vector<float> SijLocal = Sij;
            std::vector<float> PijLocal = Pij;
            std::vector<float> PVLocal = PV;
            std::vector<float> lLocal = l;
            std::vector<float> liLocal = li;
            std::vector<float> lijLocal = lij;
            std::vector<float> lnewLocal = lnew;

            for (int j = 0; j < Tc; j++) {
                int start_k = j * Bc;
                int kRows = std::min(Bc, N - start_k);

                for (int jj = 0; jj < kRows; jj++) {
                    for (int kk = 0; kk < d; kk++) {
                        int idx = start_k + jj;
                        float valK = fourDimRead(K, b, h, idx, kk, H, N, d);
                        float valV = fourDimRead(V, b, h, idx, kk, H, N, d);
                        twoDimWrite(KjLocal, jj, kk, d, valK);
                        twoDimWrite(VjLocal, jj, kk, d, valV);
                    }
                }

                for (int i = 0; i < Tr; i++) {
                    int start_q = i * Br;
                    int qRows = std::min(Br, N - start_q);

                    for (int ii = 0; ii < qRows; ii++) {
                        int row = start_q + ii;
                        liLocal[ii] = lLocal[row];
                        for (int kk = 0; kk < d; kk++) {
                            float valQ = fourDimRead(Q, b, h, row, kk, H, N, d);
                            float valO = fourDimRead(O, b, h, row, kk, H, N, d);
                            twoDimWrite(QiLocal, ii, kk, d, valQ);
                            twoDimWrite(OiLocal, ii, kk, d, valO);
                        }
                    }

                    for (int ii = 0; ii < qRows; ii++) {
                        for (int jj = 0; jj < kRows; jj++) {
                            float sum = 0.0f;
                            for (int kk = 0; kk < d; kk++) {
                                sum += twoDimRead(QiLocal, ii, kk, d) * twoDimRead(KjLocal, jj, kk, d);
                            }
                            twoDimWrite(SijLocal, ii, jj, Bc, sum);
                        }
                    }

                    for (int ii = 0; ii < qRows; ii++) {
                        float rowSum = 0.0f;
                        for (int jj = 0; jj < kRows; jj++) {
                            float val = exp(twoDimRead(SijLocal, ii, jj, Bc));
                            twoDimWrite(PijLocal, ii, jj, Bc, val);
                            rowSum += val;
                        }
                        lijLocal[ii] = rowSum;
                        lnewLocal[ii] = liLocal[ii] + lijLocal[ii];
                    }

                    for (int ii = 0; ii < qRows; ii++) {
                        for (int kk = 0; kk < d; kk++) {
                            float sum = 0.0f;
                            for (int jj = 0; jj < kRows; jj++) {
                                sum += twoDimRead(PijLocal, ii, jj, Bc) * twoDimRead(VjLocal, jj, kk, d);
                            }
                            twoDimWrite(PVLocal, ii, kk, d, sum);
                        }
                    }

                    for (int ii = 0; ii < qRows; ii++) {
                        int row = start_q + ii;
                        float invLnew = 1.0f / lnewLocal[ii];
                        lLocal[row] = lnewLocal[ii];
                        for (int kk = 0; kk < d; kk++) {
                            float updated = (liLocal[ii] * twoDimRead(OiLocal, ii, kk, d) + twoDimRead(PVLocal, ii, kk, d)) * invLnew;
                            fourDimWrite(O, b, h, row, kk, H, N, d, updated);
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
