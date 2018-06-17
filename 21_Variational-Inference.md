
$$
\newcommand{\KLDiv}{\mathbb{KL}}
\newcommand{\bE}{\mathbb{E}}
$$


# MLAPP 读书笔记 

## 21 变分推断法(Variational Inference)

> A Chinese Notes of MLAPP，MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[shouldsee](https://www.zhihu.com/people/shouldsee/activities)


### 21.1 导论

我们在本书中已经探讨了好几种计算后验分布（以及其函数）的算法。 对于离散的图模型，我们可以用链接树算法（JTA:junction tree algorithm）来进行确切(exact)的推断，前文\ref{sec:20.4}已经加以讨论。但是，这个算法的时间复杂是图链接树的宽度的指数函数，由此导致确切推断常常是不现实的。对于高斯图模型，确切推断的时间复杂度是树宽的三次函数。然而，即便这个算法在我们有很多变量的时候也会变得奇慢无比。另外，链接树算法对非高斯的连续变量以及离散/连续相混合的变量是束手无策的。 

对于某些简单共用$x->D$形式的的双节点图模型,我们可以在先验分布$p(x)$和似然函数共轭的情况下，计算其后验分布$p(x|D)$确切的闭式解（也就是说似然函数必须是一个指数族分布），更多例子见Chap5\ref{chap:5}（注意在本章节中$x$代表未知变量，而在Chap5\ref{chap:5}中我们用的是$\theta$表示未知数）。

在更一般的情况下，我们必须使用近似的推断法。在Sec8.4.1\ref{sec:8.4.1}中，我们讨论了高斯近似在非共轭的双节点中的应用(如Sec8.4.3\ref{sec:8.4.3}中在逻辑回归上的应用。)

高斯近似是简单的。但是，有一些后验分布不能用高斯分布很好地模拟。比如说，在推断多项分布的参数时，狄利克雷分布是一个更好的选择；推断离散图模型的状态变量时，分类分布是一个更好的选项。

在本章节中，我们考虑一个更加一般的、基于变分推断的确定性近似推断算法(@Jordan1998，@Jakkola&Jordan2000,@Jaakola2001，@Wainwright&Jordan2008a)。其基本理念是从易处理的分布族中选取一个近似分布$q(x)$，然后设法让这个近似分布尽可能地接近真正的后验$p^*(x)\triangleq p(x|D)$。这样以来原本的推断问题就简化成了一个最优化问题。通过放宽限制或者对目标函数再次作近似，我们可以在精度和速度之间寻找平衡。因此变分推断可以用最大后验估计（MAP）算法的速度实现贝叶斯方法的统计学优势。


在本章节中，我们考虑一个更加一般的、基于变分推断的确定性近似推断算法(@Jordan1998，@Jakkola&Jordan2000,@Jaakola2001，@Wainwright&Jordan2008a)。其基本理念是从易处理的分布族中选取一个近似分布$q(x)$，然后设法让这个近似分布尽可能地接近真正的后验$p^*(x)\triangleq p(x|D)$。这样以来原本的推断问题就简化成了一个最优化问题。通过放宽限制或者对目标函数再次作近似，我们可以在精度和速度之间寻找平衡。因此变分推断可以用最大后验估计（MAP）算法的速度实现贝叶斯方法的统计学优势。





## 21.2 变分推断法

假设$p^*(x)$使我们的真实却难以处理的分布，而$q(x)$是某个便于处理的近似分布，比如说一个多维高斯分布或者因子分解过的分布。我们假设$q$具有一些可以自由参数，并且我们可以通过优化这些参数使得$q$更加像$p^*$。我们显然可以最小化损失函数KL散度:

$$
\KLDiv)(p^*||q) = \sum_{x} p^*（x） \log \frac{p^*(x)}{q(x)}  ~~\text{(21.1)}\label{eqn:21.1}
$$

但是，这玩意非常难算，因为在分布$p^*$上求期望根据题设是难以处理的。一个自然的替代选项是最小化逆KL散度：

$$
\KLDiv)(q||p^*) = \sum_{x} q(x) \log \frac{q(x)}{p^*(x)}  ~~\text{(21.2)}\label{eqn:21.2}
$$

这个目标函数最大的优势是在分布$q$上的期望是便于计算的(通过选取适当形式的$q$)。我们会在Sec21.2.2\ref{sec:21.2.2}中讨论这两个目标函数的区别。

不幸的是，公式Eqn21.2\ref{eqn:21.2}仍然没有看起来那么好算，因为即便逐点计算$p(x|D)$也是很困难的，因为有一个正规化常数$Z=p(D)$是难以处理的。但是呢，一般来说未正规化的分布$\tilde{p}(x)\triangleq p(x,D)=p^*(x)|$是很好计算的。所以我们将目标函数改为如下:

$$
J（q） = \KLDiv(q||\tilde{p})
~~\text{(21.3)}\label{eqn:21.3}
$$

当然这个写法有点滥用记号的意思，因为$\tilde{p}$严格意义上讲并不是一个概率分布。不过无所谓，让我们带入KL散度的定义：

$$
\begin{aligned}
J(q) &= \sum_{x} q(x) \log \frac{q(x)}{\tilde{p}(x)}
~~\text{(21.4)}\label{eqn:21.4}
\\
&= \sum_{x} q(x) \log \frac{q(x)}{Z p ^ * (x)}
~~\text{(21.5)}\label{eqn:21.5}
\\ &= \sum_{x} q(x) \log \frac{q(x)}{p^*(x)} - \log Z
~~\text{(21.6)}\label{eqn:21.6}
\\
&= \KLDiv(q||p^*) - \log Z
~~\text{(21.7)}\label{eqn:21.7}
\end{aligned}
$$

因为$Z$是一个常数，所以最小化$J(q)$的同时我们也就达到了迫使 $q$ 趋近 $p^*$ 的目的。

因为KL散度总是非负的，可以看出$J(q)$是负对数似然(NLL:Negative Log Likelihood)的上界:

$$
J(q) = \KLDiv(q||p^*) - \log Z \ge -\log Z = -\log p（D）
~~\text{(21.8)}\label{eqn:21.8}
$$

换句话说，我们可以尝试最大化如下被称作能量泛函的量(@Koller&Friedman2009)。它同时也是数据似然度的下界：

$$
L(q) \triangleq -J(q) = - \KLDiv(q||p^*) + \log Z \le \log Z = \log p(D)
~~\text{(21.9)}\label{eqn:21.9}
$$

因为这个界在$q=p^*$时是紧的，可以看出变分推断法和EM算法联系之紧密(见Sec11.4.7\ref{sec:11.4.7})。


## 21.2.1 变分目标函数的其他意义

前述的目标函数还有几种同样深刻的写法。其中一种如下：

$$
J(q) = \bE_q[\log q(x) ] + \bE [-\log \tilde{p}(x)] = - \mathbb{H}(q) + \bE_q [E(x)] 
~~\text{(21.10)}\label{eqn:21.10}
$$

也就是能量的期望（因为$E(x)=-\log \tilde { p} (x)$）减去系统的熵。在统计物理里，$J(q)$被称为变分自由能，或者也叫亥姆霍兹自由能。

另一写法如下:

$$
\begin{aligned}
J(q) 
&= \bE_q{ [\log q(x) - \log p(x) p (D|x) ] }
~~\text{(21.11)}\label{eqn:21.11}
\\ 
&= \bE_q{ [\log q(x) - \log p(x) - \log p (D|x) ] }
~~\text{(21.12)}\label{eqn:21.12}
\\ 
&= \bE_q [ -\log p(D|x)  ] + \KLDiv (q(x)||p(x)) 
~~\text{(21.13)}\label{eqn:21.13}
\end{aligned}
$$

也就是负对数似然的期望加上一个表示后验分布到确切的先验距离的惩罚项。

我们还可以从信息论的角度理解(也叫作[bits-back论述](http://users.ics.aalto.fi/harri/thesis/valpola_thesis/node30.html)，具体见(@Hinton&Camp1993,@Honkela&Valpola2004))。

## 21.2.2 前向(Forward)还是后向(Reverse)KL散度？

因为KL散度是非对称的，对 $q$ 最小化 $\KLDiv(q||p)$ 和 $\KLDiv(p||q)$ 会给出不同的结果。 接下来我们讨论一下两者的异同。

首先考虑后向KL， $\KLDiv(q||p)$，也称**I-投影**或**信息投影**。根据定义有

$$
\KLDiv(q||p) = \sum_x q(x) \ln {q(x) \over p(x)}
~~\text{(21.14)}\label{eqn:21.14}
$$

这个量在 $p(x)=0$ 且 $q(x)>0$ 是无穷的。 因此如果 $p(x)=0$ 就必须要有 $q(x) = 0$。 因此后向KL被称作是**迫零的**，而且近似分布 $q$ 常常会欠估计 $p$ 的支撑集。

接下来考虑前向KL,也称**M-投影**或者**矩投影**

$$
\KLDiv(p||q) = \sum_x p(x) \ln {p(x) \over q(x)}
~~\text{(21.15)}\label{eqn:21.15}
$$

这个量在 $q(x)=0$ 且 $p(x)>0$ 是无穷的。 因此如果 $p(x)>0$ 就必须要有 $q(x) > 0$。因此前向KL被称作**避零的**，而且近似分布 $p$ 常常会过估计 $p$ 的支撑集。

两者的区别请见图Fig21.1\ref{fig:21.1}。可以发现当真实分布 $p$ 是多模态的时候，前向KL是一个很差的选择(此处假设使用了一个单模态的 $q$),因为它给出的后验的众数/平均数会落在一个低密度的区域，恰恰落在两个模态的峰值之间。在这个情况下，后向KL不仅更便于计算，也具有更好的统计性质。

\label{fig:21.1}

\label{fig:21.2}

另一个区别显示在图Fig\ref{fig:21.2}中。此处的真实分布是一个拉长的二维高斯分布而近似分布是两个一维高斯的乘积。换句话说 $p(x) = \mathcal{N}(x|\mu,\Lambda^{-1})$，且

$$
\mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}
,
\Lambda = \begin{pmatrix} 
\Lambda_{11} & \Lambda_{12}
\\   \Lambda_{21} & \Lambda_{22}
\end{pmatrix}
$$

在图Fig21.2a\ref{fig:21.2a}中我们给出了最小化后向KL $\KLDiv(q||p)$ 的结果。在这个例子中，我们可以证明有如下解

$$
\begin{aligned}
q(x) &= \mathcal{N}(x_1|m_1,\Lambda_{11}^{-1})
\mathcal{N}(x_2|m_2,\Lambda_{22}^{-1})
~~\text{(21.17)}\label{eqn:21.17}
\\ 
m1 &=  \mu_1 - \Lambda_{11}^{-1} \Lambda_{12} (m_2 - \mu_2)
~~\text{(21.18)}\label{eqn:21.18}
\\ 
m2 &=  \mu_2 - \Lambda_{22}^{-1} \Lambda_{21} (m_1 - \mu_1)
~~\text{(21.19)}\label{eqn:21.19}
\end{aligned}
$$


图Fig21.2a\ref{fig:21.2a}的结果显示我们正确地估计了平均值，但是这个近似太紧凑了：它的方差是由 $p$ 的方差最小的那个方向所决定的。事实上，通常情况下（当然也有特例 @Turner2008）在 $q$ 是乘积分布时最小化 $\KLDiv(q||p)$ 会给出一个过于置信的近似。

图Fig21.2b\ref{fig:21.2b}给出了最小化 $\KLDiv(p||q)$ 的结果。在习题Exer21.7\ref{exer:21.7}我们已经证明对一个乘积分布最小化正向KL给出的最优解正好是其真实边际分布的乘积，也就是说有

$$
q(x) = \mathcal{N}(x_1 | \mu_1,\Lambda_{11}^{-1})
\mathcal{N}(x_2 | \mu_2,\Lambda_{22}^{-1})
~~\text{(21.20)}\label{eqn:21.20}
$$

图Fig21.2b\ref{fig:21.2b}显示出这个估计是过泛的，因为它过估计了 $p$ 的支撑集。

在本章的剩余部分，以及本书接下来的大部，我们会专注于最小化后向KL $\KLDiv (q||p)$。在Sec22.5\ref{sec:22.5}对期望传播的阐述中，我们会探讨前向KL $\KLDiv{p||q}$ 在局部的优化。

## 21.2.3 另外的一些相关的度量

通过引入参数$\alpha \in \mathbb{R}$我们可以定义如下**alpha散度**：

$$
D_{\alpha}( p || q ) \triangleq \frac{4}{1-\alpha^2} 
\left( 1 - \int p(x) ^{(1+\alpha)/2} q(x) ^{(1-\alpha)/2} dx
\right)
~~\text{(21.21)}\label{eqn:21.21}
$$

这个量满足 $D_\alpha (p || q) \iff p=q$，但是它们显然也是不对称的，因而不是一个度规。 $\KLDiv(p||q)$ 对应极限 $\alpha \rightarrow 1$ ，而 $\KLDiv(q||p)$ 对应极限 $\alpha \rightarrow -1$。当 $\alpha=0$，我们取得一个和海灵格距离线性相关的对称的散度，定义如下

$$
D_H(p||q) \triangleq \int \left( p(x)^{1\over2} - q(x) ^{1\over 2}\right)^2
~~\text{(21.22)}\label{eqn:21.22}
$$

注意到 $\sqrt{D_H(p||q)}$ 是一个有效的距离度规。也就是说，它对称非负且满足三角不等式，详见(@Minka2005)。
