## 3. Hypothesis Test · 假设检验

### 3.1. Fundamental Principle · 基本原理

#### 3.1.1. Concepts · 概念

1. **Hypothesis Test**

   A part of `Statistical Inference`. Do test for the statistical hypothesis of the population with *sample data* by the designed methods, to judge whether the hypothesis is correct or not.

2. **Null Hypothesis** and **Alternative Hypothesis**

   `Null Hypothesis`: The hypothesis of population.
   `Alternative Hypothesis`: The hypothesis contrary to the null hypothesis.
   $$
   \left\{ \begin{aligned}
   & H_0: {\theta \in \Theta_0}, \quad\quad\text{Null Hypothesis}  \\
   & H_1: {\theta \in \Theta_1}, \quad\quad\text{Alternativre Hypothesis}
   \end{aligned} \right.
   $$
   where $\{\Theta_0,\Theta_1\} \in \Theta$ and $\Theta_0 \cap \Theta_1 = \varnothing$.
   $$
   \text{1-D Real Parameter Test}
   \left\{ \begin{aligned}
   &\text{(1) One-Side Test I (Left):} 	&H_0:\theta \leqslant \theta_0, H_1: \theta > \theta_0\\
   &\text{(2) One-Side Test II (Right):}	&H_0:\theta \geqslant \theta_0, H_1: \theta < \theta_0\\
   &\text{(3) Two-Side Test:}			&H_0:\theta = \theta_0, H_1: \theta \ne \theta_0
   \end{aligned} \right.
   $$

3. **Reject Domain** and **Acceptance Domain**

   **<font color=red>The direction of $W$ is as same as $H_1$.</font>**

   `Rejection Domain`: $W=\{g \in C\}$. Reject $H_0$ when the sample $g$ falls into this domain $C$.
   `Acceptance Domain`: $\{g \in C^C\}$ ($C^C$ is the *completement set* of $C$). Do not reject $H_0$ when the sample $g$ falls into this domain $C^C$ (Does not mean full acceptance).

4. **Significance Level** $\alpha$ and **Confidence Level** $1-\alpha$

   The `Significance Level` $\alpha$ is essentially a probability.

   Suppose that sample $S=\{X_1,X_2,\cdots,X_n\}$ is taken from the population $X$, and there is a domain $W \in \Omega$ ($W$ is a subset of the sample space $\Omega$). For a given $\alpha \in (0,1)$, if $W$ satisfies
   $$
   P\{ \theta(X_1,X_2,\cdots,X_n) \in W \} \leqslant \alpha, \quad\quad\forall\theta \in \Theta_0
   $$
   then the test method with $W$ as the rejection domain is called the test of significance level $\alpha$.
   *The common value of $\alpha$: 0.1, 0.05, 0.01. (Usually very small)*

   **==We use $\alpha$ as the dividing line between high Prob. and low Prob. event.==** 
   	(1) If the event Prob. $>\alpha$, then it is a high Prob. event, which means it likely to happen;
   	(2) If the event Prob. $<\alpha$, then it is a low Prob. event, which means it hardly to happen. 
   ==But if in our test, the low Prob. event did happen (Prob. $P$ of our hypothesis event is less than $\alpha$), that means the $H_0$ we set is unreliable, so we reject $H_0$.== **<font color=red>In hypothesis test, we always called the probability of the sample $g$ falls into $W$ as `p-value` denoted as $p$</font>**, so we have
   $$
   \left\{ \begin{aligned}
   & p < \alpha: \text{reject $H_0$, accept $H_1$} \\
   & p > \alpha: \text{reject $H_1$, accept $H_0$}
   \end{aligned} \right.
   $$
   So $\alpha$ is usually set to be very small to protect $H_0$, which means that in extreme cases, we have a very reliable basis to reject $H_0$ to ensure the reliability of the test.

#### 3.1.2. Step · 步骤

1. For the parameter $\theta$ to be tested, propose the one-side or two-side hyphothesis as needed;
2. Select the appropriate $\alpha$. The more strict the test, the smaller the $\alpha$. Usually $\alpha=0.05$;
3. Construct the statistic $g=g(X_1,X_2,\cdots,X_n)$ related to the hypothesis we proposed. Rejection domain $W=\{ g \in C \}$;
4. Determine $W$ by definition.

### 3.2. Parameter Test · 参数检验

Mainly to test $\mu$ and $\sigma^2$ of the normal population $\mathcal{N}\left( \mu,\sigma^2 \right)$.

#### 3.2.1. $\mu$ of Population $\sim \mathcal{N}\left( \mu,\sigma^2 \right)$ · 正态总体均值

1. **Single Population** $X$

   **(1) Two-Side Test:** $\cases{H_0: \mu = \mu_0 \\ H_1: \mu \ne \mu_0}$

   1. *$\sigma^2$ is known:*
      $$
      \begin{aligned}
      \text{Distribution Function: } & Z = \frac{\bar{X}-\mu_0}{\frac{\sigma}{\sqrt{n}}} \sim \mathcal{N}(0,1) \\
      \text{Rejection Domain: } & W = \vert Z \vert \geqslant Z_{\frac{\alpha}{2}} \Rightarrow (-\infin, -Z_{\frac{\alpha}{2}}] \cup [Z_{\frac{\alpha}{2}},+\infin)
      \end{aligned}
      \tag{3.1}\label{3.1}
      $$

   2. *$\sigma^2$ is unknown:*
      $$
      \begin{aligned}
      \text{Distribution Function: } & T = \frac{\bar{X}-\mu_0}{\frac{S}{\sqrt{n}}} \sim t(n-1) \\
      \text{Rejection Domain: } & W = \vert T \vert \geqslant t_{\frac{\alpha}{2}}(n-1) \Rightarrow (-\infin, -t_{\frac{\alpha}{2}}(n-1)] \cup [t_{\frac{\alpha}{2}}(n-1),+\infin)
      \end{aligned}
      \tag{3.2}\label{3.2}
      $$

   **(2) One-Side Test I:** $\cases{H_0: \mu \leqslant \mu_0 \\ H_1: \mu > \mu_0}$

      1. *$\sigma^2$ is known:*
   $$
   W = Z \geqslant Z_{\alpha} \Rightarrow [Z_{\alpha} ,+\infin)
   \tag{3.3}\label{3.3}
   $$

   2. *$\sigma^2$ is unknown:*
      $$
      W = T \geqslant t_{\alpha}(n-1) \Rightarrow[t_{\alpha}(n-1),+\infin)
      \tag{3.4}\label{3.4}
      $$

   **(3) One-Side Test II:** $\cases{H_0: \mu \geqslant \mu_0 \\ H_1: \mu < \mu_0}$

   1. *$\sigma^2$ is known:*

   $$
   W = Z \leqslant Z_{1-\alpha} \Rightarrow (-\infin, -Z_{\alpha}]
   \tag{3.5}\label{3.5}
   $$

   2. *$\sigma^2$ is unknown:*
      $$
      W = T \leqslant t_{1-\alpha}(n-1) \Rightarrow (-\infin, -t_{\alpha}(n-1)]
      \tag{3.6}\label{3.6}
      $$

2. **Two Population** $X,Y$

      **(1) Two-Side Test:** $\cases{H_0: \mu_1 = \mu_2 \\ H_1: \mu_1 \ne \mu_2}$

   1. *$\sigma_1^2, \sigma_2^2$ is known:*
      $$
      \begin{aligned}
      & Z=\frac{(\bar{X}-\bar{Y})-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}} \sim \mathcal{N}(0,1) \\
      & W = \vert Z \vert \geqslant Z_{\frac{\alpha}{2}} \Rightarrow (-\infin, -Z_{\frac{\alpha}{2}}] \cup [Z_{\frac{\alpha}{2}},+\infin)
      \end{aligned}
      \tag{3.7}\label{3.7}
      $$

   2. *$\sigma_1^2 = \sigma_2^2$ and unknown:*
      $$
      \begin{aligned}
      & T = \frac{(\bar{X}-\bar{Y}) - (\mu_1-\mu_2)}{S_\omega \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t_{\frac{\alpha}{2}}(n_1 + n_2 - 2) \\
      
      \text{where } & S_\omega = \sqrt{\frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}} \text{ is the combined estimator of }\sigma^2\\\\
      
      & W = \vert T \vert \geqslant t_{\frac{\alpha}{2}}(n_1 + n_2 - 2) \Rightarrow (-\infin, -t_{\frac{\alpha}{2}}(n_1 + n_2 - 2)] \cup [t_{\frac{\alpha}{2}}(n_1 + n_2 - 2),+\infin)
      \end{aligned}
      \tag{3.8}\label{3.8}
      $$
   
   3. *$\sigma_1^2 \ne \sigma_2^2$ and unknown:*
      $$
      \begin{aligned}
      & T = \frac{(\bar{X}-\bar{Y}) - (\mu_1-\mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \sim t_{\frac{\alpha}{2}}(\nu) \\\\
      
      \text{where } &
      	\left\{\begin{aligned}
      		& \nu = \frac{\left( \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2} \right)^2}{\frac{\left( \frac{\sigma_1^2}{n_1} \right)^2}{n_1-1} + \frac{\left( \frac{\sigma_2^2}{n_1} \right)^2}{n_2-1}} \\\\
      		& \hat{\nu} = \frac{\left( \frac{S_1^2}{n_2} + \frac{S_2^2}{n_2} \right)^2}{\frac{\left( \frac{S_1^2}{n_1} \right)^2}{n_1-1} + \frac{\left( \frac{S_2^2}{n_1} \right)^2}{n_2-1}}
      	\end{aligned}\right. \\\\
      	
      & W = \vert T \vert \geqslant t_{\frac{\alpha}{2}}(\nu) \Rightarrow (-\infin, -t_{\frac{\alpha}{2}}(\nu)] \cup [t_{\frac{\alpha}{2}}(\nu),+\infin)
      
      \end{aligned}
      \tag{3.9}\label{3.9}
      $$
      
   
   **(2) One-Side Test I:** $\cases{H_0: \mu_1 \leqslant \mu_2 \\ H_1: \mu_1 > \mu_2}$

      1. *$\sigma_1^2, \sigma_2^2$ is known:*
   $$
   W = Z \geqslant Z_{\alpha} \Rightarrow [Z_{\alpha} ,+\infin)
   \tag{3.10}\label{3.10}
   $$
   
   2. *$\sigma_1^2 = \sigma_2^2$ and unknown:*
      $$
      W = T \geqslant t_{\alpha}(n-1) \Rightarrow[t_{\alpha}(n-1),+\infin)
      \tag{3.11}\label{3.11}
      $$
   
   3. *$\sigma_1^2 \ne \sigma_2^2$ and unknown:*
      $$
      W = T \geqslant t_{\alpha}(\nu) \Rightarrow [t_{\alpha}(\nu),+\infin)
      \tag{3.12}\label{3.12}
      $$
   
   **(3) One-Side Test II:** $\cases{H_0: \mu_1 \geqslant \mu_2 \\ H_1: \mu_1 < \mu_2}$
   
   1. *$\sigma_1^2, \sigma_2^2$ is known:*
   
   $$
   W = Z \leqslant Z_{1-\alpha} \Rightarrow (-\infin, -Z_{\alpha}]
   \tag{3.13}\label{3.13}
   $$
   
   2. *$\sigma_1^2 = \sigma_2^2$ and unknown:*
      $$
      W = T \leqslant t_{1-\alpha}(n-1) \Rightarrow (-\infin, -t_{\alpha}(n-1)]
      \tag{3.14}\label{3.14}
      $$
   
   3. *$\sigma_1^2 \ne \sigma_2^2$ and unknown:*
      $$
      W = T \leqslant t_{\alpha}(\nu) \Rightarrow (-\infin, -t_{\alpha}(\nu)]
      \tag{3.15}\label{3.15}
      $$
      

#### 3.2.2. $\sigma^2$ of Population $\sim \mathcal{N}\left( \mu,\sigma^2 \right)$ · 正态总体方差

1. **Single Population** $X$

   **(1) Two-Side Test:** $\cases{H_0: \sigma^2 = \sigma^2_0 \\ H_1: \sigma^2 \ne \sigma^2_0}$

   1. *$\mu$ is known:*
      $$
      \begin{aligned}
      & \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (X_i-\mu)^2 \\
      
      \Rightarrow & \chi^2 = \frac{n\hat{\sigma}^2}{\sigma^2} = \frac{\sum_{i=1}^n (X_i-\mu)^2}{\sigma^2} \sim \chi^2(n) \\\\
      
      & W = \vert \chi^2 \vert \geqslant \chi^2_{\frac{\alpha}{2}}(n) \Rightarrow (-\infin, -\chi^2_{\frac{\alpha}{2}}(n)] \cup [\chi^2_{\frac{\alpha}{2}}(n),+\infin)
      \end{aligned}
      \tag{3.16}\label{3.16}
      $$

   2. *$\mu$ is unknown:*
      $$
      \begin{aligned}
      & S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i-\bar{X})^2 \\
      
      \Rightarrow & \chi^2 = \frac{(n-1)S^2}{\sigma^2} = \frac{\sum_{i=1}^n (X_i-\bar{X})^2}{\sigma^2} \sim \chi^2(n-1) \\\\
      
      & W = \vert \chi^2 \vert \geqslant \chi^2_{\frac{\alpha}{2}}(n-1) \Rightarrow (-\infin, -\chi^2_{\frac{\alpha}{2}}(n-1)] \cup [\chi^2_{\frac{\alpha}{2}}(n-1),+\infin)
      \end{aligned}
      \tag{3.17}\label{3.17}
      $$

   **(2) One-Side Test I:** $\cases{H_0: \sigma^2 \leqslant \sigma^2_0 \\ H_1: \sigma^2 > \sigma^2_0}$

         1. *$\mu$ is known:*

   $$
   W = \chi^2 \geqslant \chi^2_{\alpha}(n) \Rightarrow [\chi^2_{\alpha}(n) ,+\infin)
   \tag{3.18}\label{3.18}
   $$

   2. *$\mu$ is unknown:*
      $$
      W = \chi^2 \geqslant \chi^2_{\alpha}(n-1) \Rightarrow[\chi^2_{\alpha}(n-1), +\infin)
      \tag{3.19}\label{3.19}
      $$

   **(3) One-Side Test II:** $\cases{H_0: \sigma^2 \geqslant \sigma^2_0 \\ H_1: \sigma^2 < \sigma^2_0}$

   1. *$\mu$ is known:*

   $$
   W = \chi^2 \leqslant \chi^2_{1-\alpha}(n) \Rightarrow (-\infin, -\chi^2_{\alpha}(n)]
   \tag{3.20}\label{3.20}
   $$

   2. *$\mu$ is unknown:*
      $$
      W = \chi^2 \leqslant \chi^2_{1-\alpha}(n-1) \Rightarrow (-\infin, -\chi^2_{\alpha}(n-1)]
      \tag{3.21}\label{3.21}
      $$

2. **Two Population** $X,Y$

   1. **(1) Two-Side Test:** $\cases{H_0: \sigma^2 = \sigma^2_0 \\ H_1: \sigma^2 \ne \sigma^2_0}$

      1. *$\mu$ is known:*
         $$
         \begin{aligned}
         & F=\frac{\frac{\hat{\sigma}_1^2}{\sigma_1^2}}{\frac{\hat{\sigma}_2^2}{\sigma_2^2}} \sim F(n_1,n_2) \\ 
         
         & W = \vert F \vert \geqslant F_{\frac{\alpha}{2}}(n_1,n_2) \Rightarrow (-\infin, -F_{\frac{\alpha}{2}}(n_1,n_2)] \cup [F_{\frac{\alpha}{2}}(n_1,n_2),+\infin)
         \end{aligned}
         \tag{3.22}\label{3.22}
         $$

      2. *$\mu$ is unknown:*
         $$
         \begin{aligned}
         & F=\frac{\frac{S_1^2}{\sigma_1^2}}{\frac{S_2^2}{\sigma_2^2}} \sim F(n_1-1,n_2-1) \\ 
         
         & W = \vert F \vert \geqslant F_{\frac{\alpha}{2}}(n_1-1,n_2-1) \Rightarrow (-\infin, -F_{\frac{\alpha}{2}}(n_1-1,n_2-1)] \cup [F_{\frac{\alpha}{2}}(n_1-1,n_2-1),+\infin)
         \end{aligned}
         \tag{3.23}\label{3.23}
         $$

      **(2) One-Side Test I:** $\cases{H_0: \sigma^2 \leqslant \sigma^2_0 \\ H_1: \sigma^2 > \sigma^2_0}$

            1. *$\mu$ is known:*

      $$
      W = F \geqslant F_{\alpha}(n_1,n_2) \Rightarrow [F_{\alpha}(n_1,n_2), +\infin)
      \tag{3.24}\label{3.24}
      $$

      2. *$\mu$ is unknown:*
         $$
         W = F \geqslant F_{\alpha}(n_1-1,n_2-1) \Rightarrow [F_{\alpha}(n_1-1,n_2-1), +\infin)
         \tag{3.25}\label{3.25}
         $$

      **(3) One-Side Test II:** $\cases{H_0: \sigma^2 \geqslant \sigma^2_0 \\ H_1: \sigma^2 < \sigma^2_0}$

      1. *$\mu$ is known:*

      $$
      W = F \leqslant F_{\alpha}(n_1,n_2) \Rightarrow (-\infin, F_{\alpha}(n_1,n_2)]
      \tag{3.26}\label{3.26}
      $$

      2. *$\mu$ is unknown:*
         $$
         W = F \leqslant F_{\alpha}(n_1-1,n_2-1) \Rightarrow (-\infin, F_{\alpha}(n_1-1,n_2-1)]
         \tag{3.27}\label{3.27}
         $$

#### 3.2.3. Binomial Population · 二项分布总体

$$
\left\{ \begin{aligned}
& H_0: p = p_0 \\
& H_1: p \ne p_0
\end{aligned} \right.
$$

