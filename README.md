## 目的

目前主要用来验证平常见到的花里胡哨的“优化”。

## 方法

为了证明性能问题不是我写得不好或者踩了编译器的坑引起的，使用SIMD intrinsics或CUDA来编程，尽量将用于比较的几种算法都优化到极致。

为了说明指令执行效率所造成的差异，测试数据量不会很大，保证能放入cache中。否则，若cache缺失将成为主要性能瓶颈，则在测试结果中看不出指令执行效率的差异。