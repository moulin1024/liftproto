# liftproto：legacy 科学计算代码 → 带证据的数学结构

本原型把一个 legacy Fortran 77 代码提升为函数式 IR，并给出它的数学结构。结构中的每一条事实都带有证据等级。这一轮把三种手段结合起来：

- **静态语义分析**：确定性的提升，结论对**所有问题规模**成立，由 z3 和抽象解释证明。
- **动态分析**：在真实二进制上做逐位重放，并用 canary 检测越界。
- **LLM 工作流**：LLM 只处理前两者解决不了的事。它的所有提议都通过受限 DSL 提交，由检验器判定。

```
python liftctl.py init ws fortran/pcg.f fortran/pcgmain.f   # 静态提升 + 动态验证 + 证明 + 覆盖率，几秒内完成
python liftctl.py ws worklist                               # LLM 的待办义务列表
python liftctl.py ws <tool> '{json}'                        # LLM 通过工具提交提议（见 AGENT_GUIDE.md）
python liftctl.py ws structure                              # 输出最终结构文档 ws/STRUCTURE.md
claude mcp add lift -- python mcp_server.py ws              # 或作为 MCP server 接入 Claude Code 等 agent
python replay.py OLD_WS NEW_WS                              # 在新工作区重放一次 agent 会话，会话本身成为可复现的证据
python tests/test_agent_channel.py                          # 对 LLM 通道做红队测试
python structdiff.py                                        # 静态结构 diff：判定候选优化（第一轮的 7 个变异体）
```

依赖：numpy、gfortran、gcov、z3-solver、mcp。代码总量约 4500 行 Python。

## 1. 分工

```
           ┌─────────────── 可信域（确定性） ───────────────┐
 legacy.f ─┤ 静态分析 ─ lifter ─ z3 证明 ─┐                  │
           │                              ├─ 事实库（带证据）─ STRUCTURE.md
 binary  ──┤ 动态探测 ─ 逐位重放 ─ gcov ──┘        │          │
           └───────────────────────────────────────┼──────────┘
                                         worklist  │  ▲ 提议（DSL / JSON，不执行）
                                                   ▼  │
                                                 LLM agent
```

| 层 | 负责 | 给出的证据 |
|---|---|---|
| 静态 `lift/static/` | 提升 map、fold、2-D stencil、手工展开、前向 GOTO、提前 RETURN、block IF；把驱动程序提升为 let 序列加 while 循环（通过循环旋转）；切分手工划分的 work 数组；推导接口契约和探测规约 | 所有规模下成立（z3 / 引理） |
| 动态 `lift/verify.py` 等 | 保序逐位重放；写集和读集、次数、形状；精确 Jacobian；差分 canary；gcov 覆盖率 | 2 个可见规模加 1 个检验器私有规模 |
| LLM（`AGENT_GUIDE.md`） | 不透明单元的 spec、term 或 primitive 契约；针对未覆盖分支的定向探测；算法模板识别；物理解释 | 只提出，由检验器给出证据；纯解释性内容记为 agent-claim |

## 2. 测试代码 `fortran/pcg.f`

它在第一轮的 PCG 基础上，把求范数换成了参考 BLAS 的 **DNRM2**（Hammarling 的缩放平方和：SCALE 和 SSQ 跨迭代携带，块状 IF 内有数据相关的更新）。这类代码静态提升器处理不了，必须由 LLM 来处理。

## 3. 静态分析自动得到的结果（不需要 LLM）

- **8 个单元中 7 个被自动提升**，并全部通过动态逐位验证，其中包括 SETUP（含 sin、cos，与 libm 逐位一致）和整个 CGSOLV 驱动。DNRM2 被判为不透明，给出的原因是："line 202: loop-carried state (SCALE, SSQ) with an update that is not a sequential sum"。分析器先根据调用点常量把 DNRM2 特化为 `INCX=1` 再尝试提升，所以这个原因比"步长非常数"更具体。
- **DAXPY 的 `DA==0` 分支被自动保留**，提升为 `(DA == 0 ? DY : DY + DA * DX)`，并证明它只影响 IEEE 语义：用 z3 证明它在实数上与另一分支相等，所以次数仍为 1。第一轮中，这个 signed-zero 差异要靠 gcov 加人工定向探测才发现。
- **手工展开**（DDOT 展开 5 次、DAXPY 展开 4 次）被证明恰好覆盖 `[1..N]`（z3，带 Fortran 的 MOD 语义），并还原成普通的 `foldl` 或 map。第一轮里 gcov 报告的两个"零次迭代分支"被**证明不可达**，覆盖率缺口因此由静态证明关闭。
- **work 数组切分**：`W = R[1..N] ++ Z[N+1..2N] ++ P[2N+1..3N] ++ Q[3N+1..4N]`，已证明不相交且恰好铺满 `[1..4N]`，段名取自偏移变量名（IR→R 等）。
- **发现隐含的接口契约**：AMUL 不使用它的参数 N（编译器只给出 unused 警告），实际按 COMMON 中的 NX·NY 访问。分析器由此推出 `NX*NY == N`，并在 MAIN 的调用点核验成立。另一个契约是 `INCX == 1`，在所有调用点都成立。
- **算子性质**（AMUL）：精确线性分解为 5 点 stencil；**IEEE 意义下严格对称**，并证明对所有规模成立，方法是检验 `c_s(p) == c_-s(p+s)`，只用 + 和 * 的交换律（IEEE 下精确成立），不用结合律；PCOND 被证明是 **Jacobi 预条件子**，即 `diag(A) = DG`；**SPD（实数意义，所有 NX、NY ≥ 1）**：区间算术给出 CK ∈ [0.5, 1.5]，z3 非线性实数算术证明对角为正、处处弱对角占优、边界处严格占优，再加上对称性和网格图的连通性。
- **静态推导出动态探测的 spec**：数组长度来自足迹分析，整数参数来自调用点常量或网格尺寸，SETUP 的前置调用来自 MAIN 的调用顺序。第一轮中这部分需要 agent 手写。

## 4. LLM 工作流的一次真实会话

由我（Claude）扮演 agent，只通过 `liftctl.py` 与工具交互。完整记录见 `ws_demo/log.jsonl` 和 `ws_demo/STRUCTURE.md` 的第 7 节。这次会话已用 `replay.py` 在最终版代码上重放，结果相同。

| 步骤 | agent 的提议 | 检验器的判定 |
|---|---|---|
| 1 | DNRM2 的 spec：`X: N`，`INCX: 1` | 接受（差分 canary 通过） |
| 2 | DNRM2 的 term：`sqrt(foldl(+, 0, X * X))` | **拒绝**：只在实数意义上等价（相对误差 4e-16），不逐位相等，并提示改为 primitive |
| 3 | 把 DNRM2 注册为 primitive，契约为上式，性质为 `nonnegative`、`homogeneous:1`，tol 1e-14 | 接受：30 个探测，误差 4e-16，**正齐次性逐位成立**；之后用原生代码重放，驱动程序仍逐位一致 |
| 4–12 | 9 个定向探测：DAXPY 的 `DA=0` 加 `-0.0`、`N=3`、`N=0`；DDOT 的 `N=0`、`N=3`；DNRM2 的 `N=0`、`N=1`、含零的 X；CGSOLV 的 `MAXIT=3` | 全部一致，gcov 缺口全部关闭 |
| 13 | DNRM2 第 202 行零次迭代分支不可达的论证 | 记录为**未验证的 agent-claim**（DNRM2 是不透明单元，无法静态证明） |
| 14 | PCG 模板（Saad 算法 9.1），写成 `r = b - A(x)` | **不匹配**："body: 8 vs 9 statements" |
| 15 | 改为 `q = A(x); r = b - q` | **α-等价匹配**（15 个名字对应），事实为 "CGSOLV implements PCG with A=AMUL, Minv=PCOND, …" |
| 16–17 | AMUL 的物理解释：−∇·(k∇u)，算术平均面导率，Dirichlet 边界；SETUP 的 k(x, y) | 记录为 agent-claim |

**最终结果**：worklist 清空，共 48 条事实。其中 40 条为 ESTABLISHED（静态加动态），3 条为 PROVED（仅静态），2 条为 TESTED（仅动态，属于 DNRM2），3 条为 CLAIMED（agent 的解释）。结构文档见 `ws_demo/STRUCTURE.md`。

## 5. LLM 通道的红队测试（`tests/test_agent_channel.py`，12/12 通过）

覆盖的情形包括：代码注入（spec 表达式、DSL）、错误的数学（L1 范数冒充 L2）、错误的性质（homogeneous:2）、自引用契约、过宽的容差、错误的算法模板、超大探测规模，以及长度给短的 spec。

**红队测试发现了一个真实漏洞**：把 DNRM2 的 spec 给成 `X: N-1` 时，最初被**接受**了。原因是 canary 只能检测越界写，越界读到的 NaN 混进了输出却没有被识别。修复方法是**差分 canary**：把每个数组两侧的填充分别换成有限的诱饵值，再重跑一遍；只要任何输出发生变化，就判定为越界读。修复后这类 spec 会被拒绝，ladder 里也会对每个规模做一次这项检查。

## 6. 静态结构 diff：判定候选优化（`structdiff.py`，对象是第一轮的变异体）

| 候选 | 静态判定（所有规模） | 第一轮的动态判定 |
|---|---|---|
| daxpy_plain | ACCEPT，license: compiler（提升结果相同） | 相同 |
| pcond_recip | ACCEPT?，license: statistical（z3 在分母非零时证明实数等价） | 相同 |
| amul_asym | REJECT（语义改变，对称性证明失败） | 相同 |
| ddot_drop | REJECT（z3 给出**反例 N=6**：分段铺不满 `[1..N]`） | 相同 |
| dcopy_overrun | REJECT（足迹从 N 变为 N+1） | 相同 |
| ddot_truncate_large | 无法静态提升（循环体中有跳出的 GOTO），需要动态证据 | 只有私有规模能拦住 |
| ddot_two_acc | 无法静态提升（双累加器），需要动态证据 | statistical |

静态和动态是互补的：静态能处理的部分，结论对所有规模成立，还能给出反例；静态处理不了的部分（明确标为 "NEEDS EVIDENCE"），交给动态检验和 LLM。

## 7. 局限

- 静态提升器只支持 F77 的一个子集：仿射下标、至多 2 层完美嵌套、常数步长、前向 GOTO、单个 while 模式的驱动、单输出子程序。不支持 MPI、EQUIVALENCE、间接寻址、条件写。超出范围的部分会明确报为 opaque，不会猜测。
- 实数意义的证明（SPD）不包含舍入误差。IEEE 严格的证明只用了交换律。
- spec 给得过长目前检测不到。agent 的不可达论证不是证明。动态证据只覆盖实际测试过的规模。
- agent 的物理解释（annotate）不经验证，只作为 agent-claim 出现在结构文档里。

---

# 第一轮：纯动态原型（`fortran/cg.f`、`run_lift.py`、`mutate.py`）

用一个故意写成 F77 风格的 Jacobi 预条件 CG（`fortran/cg.f`）测试"动态分析 + agent 提出 + 确定性检验"的提升框架。框架用 Python 实现（约 700 行），只依赖 numpy、gfortran 和 gcov。

```
bash run_all.sh        # 提升 + 证书 + 变异测试 + 编译器对照基准，约 5 秒
```

## 目录

| 路径 | 作用 | 可信？ |
|---|---|---|
| `fortran/cg.f` | 被测 legacy 代码：IMPLICIT、COMMON、手工划分的 work 数组、GOTO 循环、BLAS 式手工展开 | — |
| `lift/f77.py` | 只解析**接口**（参数、类型、COMMON 布局），不分析可执行语句 | ✔ |
| `lift/harness.py` | 编译成 .so，ctypes 调用；COMMON 映射为 numpy 视图；每个数组参数两侧放**按参数和方向编码**的 NaN canary；gcov 解析 | ✔ |
| `lift/probe.py` | 写集、读集、多项式次数（有限差分）、精确齐次性（2 的幂缩放，逐位）、元素级依赖关系 | ✔ |
| `lift/shape.py` | 依赖关系 → 组合子形状（map / fold / stencil2d / general） | ✔ |
| `lift/ir.py` | 组合子 IR，加上**保序**解释器（归约是显式的顺序 foldl；`np.sum` 用的是 pairwise 求和，不能逐位重放） | ✔ |
| `lift/verify.py` | 验证阶梯 V1–V3、算子系数提取、证书 | ✔ |
| `hypotheses/cg_agent.py` | **agent 的输出**：数组长度、提升出的项、系数解释、定向探测、修订 | ✘ 不可信 |
| `run_lift.py` | 编排：probe → cov1 → round2 → cov2 → certificates.json | ✔ |
| `mutate.py` | 把"候选优化"当作变异体，对照参考结构判定 | ✔ |
| `bench/` | 手工展开与普通循环在各优化级别下的对照 | — |

## 结果

### 1. 自动推断出的结构事实（不需要 agent）

| 单元 | 写集 | 读集 | 次数 | 形状 |
|---|---|---|---|---|
| DDOT | result | DX, DY | 双线性 | fold over N |
| DAXPY | DY | DA, DX, DY | 各自 1 次（DA·DX 双线性） | map |
| PCOND | Z | R, GEOM.DG | R: 1 次；DG: 非多项式，齐次度 −1（逐位） | map |
| AMUL | Y | X, HX2I, HY2I, CK, DG | X: 线性，齐次度 1（逐位） | **stencil2d，5 点，截断边界（齐次 Dirichlet）** |
| CGSOLV | X, W, ITER, RNORM | B, X, TOL | — | — |

AMUL 的 5 点结构和网格形状 (NX, NY) 是从依赖关系里**推断出来的**，代码里只有线性化下标 `K=(J-1)*NX+I`。

### 2. 提升出的项（节选）

```
DDOT  = λ(DX, DY). foldl(+, 0, DX * DY)            -- 手工展开 5 被还原成普通 fold
DAXPY = λ(DA, DX, DY). (DA == 0 ? DY : DY + DA * DX)   -- 第二轮修订后
AMUL  = λ(X). let S0 = DG * X
              let S1 = (i > 1  ? S0 - HX2I*0.5*(CK + CK[i-1,j]) * X[i-1,j] : S0) ...
CGSOLV = λ(B, X, TOL). ... while_loop(...) ... in {X, W = [R ++ Z ++ P ++ Q], ITER, RNORM}
```
GOTO 循环提升为 `while_loop`；手工划分的 `W(4N)` 被识别为 4 个具名向量的拼接，这一点经逐位比较 W 得到验证。

### 3. 验证阶梯

在 3 个规模上测试：两个 agent 可见的 (7,5)、(6,8)，以及一个检验器私有的 (16,14)。

- **全部 7 个单元，包括整个 CG 求解（27–64 次迭代）：V3 保序逐位重放全部通过。** 也就是说，纯 IR 解释器算出的 X、W、ITER、RNORM 与 Fortran 的结果**逐位相同**。
- -O2 构建和 -O0 构建（coverage）下所有探测输出的摘要完全一致，即结果不随编译选项变化。
- AMUL 算子：通过 delta 探测精确提取出系数场，agent 给出的系数解释在所有点上逐位成立；**A 严格对称**（对任意随机系数场都成立，所以这是结构性质）；在 SETUP 几何下对角为正，且弱对角占优（误差在舍入量级）并存在严格占优的行 ⇒ **SPD（实数意义下）**。这正是 CG 依赖的 L0 不变量。

### 4. agent 循环示例（V4 覆盖率驱动）

第一轮 gcov 报告了 DDOT 和 DAXPY 中几个未走到的分支：`N≤0`、`N<5`、`DA.EQ.0`。agent 针对这些分支提出定向探测后：

- `DA = 0` 且 `DY` 中含 `-0.0` 时，第一轮的项 `DY + DA*DX` 给出 `+0.0`，Fortran 给出 `-0.0` → **反驳**。
- agent 修订为 `(DA == 0 ? DY : …)`，修订后全部通过。驱动程序换用修订后的 DAXPY，仍然逐位一致。
- 剩下唯一的覆盖缺口是 `DO 50` 的零次迭代分支。它实际上**不可达**（N<5 时已经跳过这个循环），但动态分析分辨不出"不可达"和"没测到"，这是需要轻量静态推理（路径条件 + SMT）来补上的地方。

### 5. 变异测试：候选优化怎么判

| 候选 | 描述 | 判定 | 被哪一层拦住 |
|---|---|---|---|
| daxpy_plain | 去掉手工展开 | ACCEPT，license: compiler | 逐位相同，说明没有语义内容 |
| ddot_two_acc | 双累加器（重结合） | ACCEPT?，license: statistical | DDOT 实数等价但浮点不同；驱动轨迹不同，但满足后置条件 |
| pcond_recip | `R*(1/DG)` | ACCEPT?，license: statistical | 同上 |
| amul_asym | 西侧系数写错 | REJECT | AMUL 被反驳，A 不对称，残差 6e-2 |
| ddot_drop | 余数循环少算一个元素 | REJECT | **只有 kernel 层拦得住**，见下 |
| ddot_truncate_large | N>150 时少算（reward hack） | REJECT | **只有私有规模 (16,14) 拦得住** |
| dcopy_overrun | 多写一个元素 | REJECT | canary（越界） |

**两个关键发现：**

1. **只看端到端后置条件不够。** `ddot_drop` 和 `ddot_truncate_large` 让 CG 仍然收敛到 `‖b−Ax‖/‖b‖ < 1e-9`，因为 CG 有自我修正能力；代价只体现在迭代次数上（64 → 86 和 64 → 251）。如果只用"结果对不对"做 oracle，这两个都会被接受。
2. 第一版 canary 对所有参数用的是同一个 NaN 模式，`dcopy_overrun` 把 DX 的 canary 原样复制进了 DY 的 pad，结果**骗过了检验**。改成按参数和方向编码 payload 之后才拦住。这正是一个现成的 reward-hacking 教训：检验器的"签名"必须让被测代码无法伪造。

### 6. 编译器对照（`bench/results.txt`，N=2e6，重复 60 次，取 3 次中最好）

| flags | ddot 展开 | ddot 普通 | daxpy 展开 | daxpy 普通 |
|---|---|---|---|---|
| -O0 | 0.125 | 0.261 | 0.166 | 0.176 |
| -O2 | 0.072 | 0.074 | 0.068 | 0.069 |
| -O3 | 0.072 | 0.072 | 0.068 | 0.068 |
| -O3 -march=native | 0.069 | 0.070 | 0.070 | 0.068 |

（这是在 2 核共享容器上跑的某一次结果，重复运行之间有 ±15% 的波动，`results.txt` 里是最近一次运行的数字。）手工展开只在 -O0 下有用；从 -O2 开始，两种写法的差别在噪声范围内，结果也逐位相同。所以 lifting 把展开还原掉不会有任何损失，这类变换的 license 应该归给编译器。

附带一个教训：第一版基准把普通版和主程序放在同一个文件里，编译器把纯函数调用提到了循环外面，"普通版"看起来快了 3.6 倍。**计时 harness 本身也必须放在可信域里，并且单独编译。**

## 局限

- 接口规约（数组长度、整数参数）由 agent 提出，canary 只能检查长度"不小于实际访问"。长度给大了检测不到。
- 次数检验和读集检验是概率性的（随机方向）；对多项式部分有 Schwartz–Zippel 意义下的保证，对 ⊤（非多项式）部分只能给出"非多项式"这一结论。
- 只处理单进程；MPI、真正的间接寻址（`gather`）、FFT 和线性求解器的识别都还没做。
- 动态覆盖率无法区分不可达分支和未测分支。
