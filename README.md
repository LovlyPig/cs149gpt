
# NanoGPT149
***The original hand out is [here](https://github.com/stanford-cs149/cs149gpt)***

# Performance
## Part 1
<img width="1462" height="995" alt="image" src="https://github.com/user-attachments/assets/29533867-d91a-4ef7-945c-b6296f4818f7" />

## part 2
<img width="1642" height="995" alt="image" src="https://github.com/user-attachments/assets/932cc68e-6534-4a21-8fda-d73ede2ce4cf" />

## Part 3
<img width="1463" height="995" alt="image" src="https://github.com/user-attachments/assets/89b0ff49-d241-46ac-8264-afe9ee25d4d6" />

## Part 4

>Notice that the performance of Part 4 is slower than that of the previous parts. Have we fully optimized Part 4? What other performance improvements can be done? Please list them and >describe why they would increase performance.

In my implementation, I achieved performance exceeding the reference time (about 8–9×). I believe the slower performance of the naive FlashAttention algorithm is due to too many for-loop computations, and the use of OpenMP requiring many temporary arrays. Compared to the row‑by‑row computation approach in part 3, this collectively leads to lower cache hit rates. However, I followed the algorithm diagram very directly and did not degrade performance because of that.
Perhaps for the final step—'write blocks Oi and lnew back to O and l in main memory'—I merged this step into the final computation.

<img width="1463" height="903" alt="image" src="https://github.com/user-attachments/assets/d78e40f9-a3f6-42fd-a967-cd36edb4695d" />


