## 4. Regression Analysis · 回归分析

Through a large number of experimental observations or data, find the relationship and statistical law among the variables by statistic methods.
$$
\text{Variable Type} \left\{ \begin{aligned}
& \text{Dependent Variable: } Y \\
& \text{Independent Variable: } X_1,X_2,\cdots, X_p
\end{aligned} \right.
$$
where $Y$ also named `Response Variable`, which can represent the problems we consider in actual situations; $\{X_1,X_2,\cdots, X_p\}$ also named `Observation Variable` or `Explanatory Variable` or `Factor`, which can influence the value of $Y$.

**Main Problems:**
(1) Determine the relational expression of $\{X_1,X_2,\cdots, X_p\}$ (*i.e.*, the coefficients of $X_i$), that is, determine the `Regression Function`;
(2) Test the credibility degree of the regression function;
(3) Judge whether $\{X_1,X_2,\cdots, X_p\}$ has an effect on $Y$;
(4) Make prediction and control by the obtained regression function.

**==Note:==**
In a linear regression model, there are $p$ kinds of independent variable $\{X_1,X_2,\cdots, X_p\}$; and $n$ sample numbers of each independent variable like $\{X_{11},X_{21},\cdots,X_{n1}\}$. All independent variables can be show as the follow matrix (called *design matrix* in multiple linear regression):
$$
X_{n \times (p+1)} = 
\begin{bmatrix}
X_{11}	&	X_{21}	&	\cdots	&	X_{n1}	\\
X_{12}	&	X_{22}	&	\cdots	&	X_{n2}	\\
\vdots	&	\vdots	&	\ddots	&	\vdots	\\
X_{1p}	&	X_{2p}	&	\cdots	&	X_{np} 
\end{bmatrix}
$$

### 4.1. Simple Linear Regression · 一元线性回归

$$
\left\{ \begin{aligned}
& \text{Simple (Variable): Only have one independent variable } X \\
& \text{Linear: The relationship between $Y$ and $X$ is linear} 
\end{aligned} \right. \\
Y = \beta_0+\beta_1X+\epsilon
$$

where: 
	(1) $\beta_0+\beta_1X$ represents the part where $Y$ varies linearly with the change of $X$. $f(x)=\beta_0+\beta_1X$ is called `Simple Linear Function`, where $\beta_0$ is `Regression Constant`, $\beta_1$ is `Regression Coeeficient` (Collectively known as `Regression Parameter`);
	(2) $\epsilon$ is the `Random Error` that means the sum of the influence of all uncertain factors. The value of $\epsilon$ cannot be observation, it can only be obtained when the estimated value is fitted with the real value **(*Sample* `Residual` $r$ is the estimate of *Population* `Random Error` $\epsilon$)**. We usually suppose $\epsilon \sim \mathcal{N}\left( 0, \sigma^2 \right)$. If $\epsilon$ does not satisfy this, it shows that the problem is not suitable to be analysis by regression analysis.

If $\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\}$ is a group of observations of $(X,Y)$, then the `Simple Linear Regression Model` is
$$
\begin{aligned}
& Y = \beta_0+\beta_1X+\epsilon \\
& y_i = \beta_0 + \beta_1x_i + \epsilon_i, \quad\quad i=1, 2, \cdots, n
\end{aligned}
\tag{4.1}\label{4.1}
$$
where $\mathbb{E}(\epsilon) = 0$, $\text{Var}(\epsilon) = \sigma^2$(`Homogeneity of Variance` 方差齐性: The value of Var. is fixed). The $df$ of SLRM is the number of kinds of independent variables.

#### 4.1.1. OLS Estimation of Regression Parameters · 回归参数的最小二乘估计

The smaller the deviation between the point $(x_i,y_i)$ above or under the regression line and the point $(x_i,\hat{y_i})$ on the regression line, the better the regression line fits the real problem.

==fig==

where $\hat{y_i} = \hat{\beta_0}+\hat{\beta_1x_i}$ is called `Regression Value` or `Fitted Value`; $R^2= \left(y_i - \hat{y_i} \right)^2$ is called `Residual` (*p.s.*, The reason for using the square in $r$ is that the deviation values maybe position or negative, when accumulate all $r$s it perhaps be correspondingly offset and reduce the degree of deviation. In order to accurately reflect the deviation, we usually use the square.).

Let $Q(\beta_0, \beta_1) = \sum\limits_{i=1}^{n}\left(y_i - \hat{y_i} \right)^2 = \sum\limits_{i=1}^{n}\left(y_i - \beta_0-\beta_1x_i\right)^2$, then the `Ordinary Least Square Estimate` of $\beta_0,\beta_1$ is establish the following optimization problem:
$$
\begin{aligned}
& Q(\hat{\beta_0}, \hat{\beta_1}) \\\\
\Rightarrow & \mathcal{P}: \min_{\beta_0,\beta_1}Q(\beta_0, \beta_1)
\end{aligned}
$$
Calculate that
$$
\left\{ \begin{aligned}
& \hat{\beta_1} = \frac{\sum\limits_{i=1}^{n}\left[\left(x_i - \bar{x_i} \right)\left(y_i - \bar{y_i} \right)\right]}{\sum\limits_{i=1}^{n}\left(x_i - \bar{x_i} \right)^2} = \frac{S_{xy}}{S_{xx}} \\
& \hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}
\end{aligned} \right.
\tag{4.2}\label{4.2}
$$
where $\cases{\bar{x} = \frac{1}{n}\sum{x_i} \\ \bar{y} = \frac{1}{n}\sum{y_i}}$, $\cases{S_{xx} = \sum\left(x_i-\bar{x}\right)^2 \\ S_{xy} = \sum{\left(x_i - \bar{x_i} \right)\left(y_i - \bar{y_i} \right)} = \sum{y_i\left(x_i - \bar{x}\right)}}$. And we usually take the Var. of `Residual`
$$
\hat{\sigma}^2 = \frac{1}{n-2}\sum_{i=1}^{n}\left(y_i - \hat{\beta_0} - \hat{\beta_1}x_i \right)^2
\tag{4.3}\label{4.3}
$$
as the `OLS Unbiased Estimate` of the Var. of `Random Error` $\sigma^2$ (*i.e.*, $\mathbb{E}(\hat{\sigma}^2) = \sigma^2$). Where the $n-2$ is $df$ of residual, which is calculate by ==$\text{Sample (numbers - number of kind of X ($df$ of the model) - number of intercept ($\beta_0$)})$==.

And the estimate Var. of $\beta_0,\beta_1$ is as follow:

1. **$\sigma^2$ is known:**
   $$
   \left\{ \begin{aligned}
   & \text{Var}(\beta_0) = \sigma^2\left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right) \\
   & \text{Var}(\beta_1) =\frac{\sigma^2}{S_{xx}}
   \end{aligned} \right.
   \tag{4.4}\label{4.4}
   $$

2. **$\sigma^2$ is unknown:** replace $\sigma$ with $\hat{\sigma}$
   $$
   \left\{ \begin{aligned}
   & \text{SD}(\hat{\beta_0}) = \hat{\sigma} \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}} \\
   & \text{SD}(\hat{\beta_1}) =\frac{\hat{\sigma}}{\sqrt{S_{xx}}}
   \end{aligned} \right.
   \tag{4.5}\label{4.5}
   $$
   Notice that in regression analysis we usually call the SD as *Standard Error* (just a different name).

> **Derivation of the estimation of  regression parameters.**
>
> $$
> \begin{aligned}
> & Q(\beta_0, \beta_1) = \sum\limits_{i=1}^{n}\left(y_i - \beta_0-\beta_1x_i\right)^2 \\
> 
> \Rightarrow & \text{Let} \left\{ \begin{align}
> \frac{\part{Q}}{\part{\beta_0}} \Bigg|_{\hat{\beta_0},\hat{\beta_1}} & = \sum_{i=1}^{n}2(y_i-\beta_0-\beta_1x_i) \cdot (0-1-0) \\
> 	& = -2\sum_{i=1}^{n}(y_i-\hat{\beta_0}-\hat{\beta_1}x_i) = 0 \tag{1}\\
> 	
> \frac{\part{Q}}{\part{\beta_1}} \Bigg|_{\hat{\beta_0},\hat{\beta_1}} & = \sum_{i=1}^{n}2(y_i-\beta_0-\beta_1x_i) \cdot (0-0-1 \cdot x_i) \\
> 	& = \sum_{i=1}^{n}(y_i-\hat{\beta_0}-\hat{\beta_1}x_i)x_i = 0 \tag{2}
> \end{align} \right. \\
> 
> \xRightarrow{\text{Simplify (1)}} & \hat{\beta_0} = \frac{\sum{y}}{n} - \frac{\hat{\beta_1}\sum{x}}{n} = \bar{y} - \hat{\beta_1}\bar{x} \\
> 
> \xRightarrow{\text{Substitude in (2) and simplify}} & \hat{\beta_1} = \frac{\sum{y_ix_i} - \frac{\sum{y_i}\sum{x_i}}{n}}{\sum{x_i^2} - \frac{\left( \sum{x_i} \right)^2}{n}} = \frac{\sum{y_i\left( x_i-\bar{x} \right)}}{\sum{\left( x_i-\bar{x} \right)^2}} = \frac{S_{xy}}{S_{xx}}
> \end{aligned}
> $$
> *p.s.*, Proof $S_{xy} = \sum{\left(x_i - \bar{x_i} \right)\left(y_i - \bar{y_i} \right)} = \sum{y_i\left(x_i - \bar{x}\right)}$.
>
> As a constant, $\sum\bar{x},\sum\bar{y} = n\bar{x}, n\bar{y}$. 
> $$
> \begin{aligned}
> S_{xy} & = \sum{\left(x_i - \bar{x_i} \right)\left(y_i - \bar{y_i} \right)} \\
>        & = \sum{x_iy_i} - \sum{\bar{x}y_i} - \sum{\bar{y}x_i} + \sum{\bar{x}\bar{y}} \\
>        & = \sum{y_i\left(x_i - \bar{x}\right)} \color{red}{\cancel{- n\bar{y}\frac{\sum{x_i}}{n}} \cancel{+ n\bar{x}\bar{y}}} \\
>        & = \sum{y_i\left(x_i - \bar{x}\right)}
> \end{aligned}
> $$

#### 4.1.2. Significance Test of Regression Parameters · 回归参数的显著性检验

It is mainly test for $\beta_1$: $\cases{H_0: \beta_1 = 0 \\ H_1: \beta_1 \ne 0}$. If want to test $\beta_0$, we can replace $\beta_1$ with $\beta_0$ directly.

1. **t Test**
   $$
   \begin{aligned}
   & T = \frac{\hat{\beta_1}}{\text{SD}\left( \hat{\beta_1} \right)} = \frac{\hat{\beta_1}}{\frac{\hat{\sigma}}{\sqrt{S_{xx}}}} = \frac{\hat{\beta_1}\sqrt{S_{xx}}}{\hat{\sigma}} \sim t(n-p-1) = t(n-2) \\
   & W = \vert T \vert \geqslant t_{\frac{\alpha}{2}}(n-2)
   \end{aligned}
   \tag{4.6}\label{4.6}
   $$

2. **F Test**
   $$
   \begin{aligned}
   & F = \frac{\hat{\beta_1}^2S_{xx}}{\hat{\sigma}^2} \sim F(p, n-p-1)=F(1,n-2) \\
   & W = \vert F \vert \geqslant F_{\frac{\alpha}{2}}(1, n-2)
   \end{aligned}
   \tag{4.7}\label{4.7}
   $$

#### 4.1.3. Interval Estimation of Regression Parameters · 回归参数的区间估计

$$
\begin{aligned}
& T_i = \frac{\hat{\beta_i}-\beta_i}{\text{SD}\left( \hat{\beta_i} \right)} \sim t(n-p-1)=t(n-2), \quad\quad i=0,1 \\
& P \left\{\left\vert \frac{\hat{\beta_i}-\beta_i}{\text{SD}\left( \hat{\beta_i} \right)} \right\vert \leqslant t_{\frac{\alpha}{2}}(n-2)\right\} = \alpha \\\\
& \left[ \hat{\beta_i} - \text{SD}\left( \hat{\beta_i} \right)t_{\frac{\alpha}{2}}(n-2), \hat{\beta_i} + \text{SD}\left( \hat{\beta_i} \right)t_{\frac{\alpha}{2}}(n-2) \right]
\end{aligned}
\tag{4.8}\label{4.8}
$$

#### 4.1.4. Coefficient of Determination $R^2$ and Adjusted $R^2$ · 判定系数 $R^2$ 与调整后 $R^2$

| Kinds of Sum of Square                              | Formula                                                      |
| --------------------------------------------------- | ------------------------------------------------------------ |
| `Total Sum of Square` (SST, 总离差平方和)           | $SST = \sum\limits_{i=1}^n{\left(y_i - \bar{y}\right)^2} = SSE+SSR$ |
| `Sum of Square Error` (SSE, 残差平方和)             | $SSE = \sum\limits_{i=1}^n{\left(y_i - \hat{y_i}\right)^2} = \sum\limits_{i=1}^n\hat{\sigma}^2$ |
| `Sum of Square of the Regression` (SSR, 回归平方和) | $SSR = \sum\limits_{i=1}^n{\left(\hat{y_i} - \bar{y}\right)^2}$ |

$$
\left\{\begin{aligned}
& R^2 = \frac{SSE}{SST} = 1-\frac{SSR}{SST} \\\\
& R^2(adj) = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
\end{aligned}\right.
\tag{4.9}\label{4.9}
$$

where $n$ is the number of samples and $p$ is the number of independent variables.

#### 4.1.5. AIC and BIC · 赤池信息准测与贝叶斯信息准则



### 4.2. Multiple Linear Regression · 多元线性回归

`Multiple Linear Regression Function`:
$$
f(x) = \beta_0 + \beta_1X_1 + \cdots + \beta_pXp
$$
`Multiple Linear Regression Model`:
$$
\begin{aligned}
& Y = \beta_0 + \beta_1X_1 + \cdots + \beta_pXp + \epsilon \\ 
& y_i = \beta_0 + \beta_1x_i1 + \cdots + \beta_px_ip + \epsilon_i, \quad\quad i=1,2,\cdots,n
\end{aligned}
\tag{4.10}\label{4.10}
$$
The matrix form of MLEM is as follow
$$
Y = X\beta + \epsilon
\tag{4.11}\label{4.11}
$$
where
$$
Y_{n \times 1} = \begin{bmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{bmatrix},\quad

\beta_{(p + 1) \times 1} = \begin{bmatrix}
\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p
\end{bmatrix},\quad

X_{n \times (p+1)} = 
\begin{bmatrix}
X_{11}	&	X_{21}	&	\cdots	&	X_{n1}	\\
X_{12}	&	X_{22}	&	\cdots	&	X_{n2}	\\
\vdots	&	\vdots	&	\ddots	&	\vdots	\\
X_{1p}	&	X_{2p}	&	\cdots	&	X_{np} 
\end{bmatrix},\quad

\epsilon_{n \times 1}  = \begin{bmatrix}
\epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n
\end{bmatrix}
$$
where $\mathbb{E}(\epsilon) = 0$, $\text{Var}(\epsilon) = \sigma^2 \cdot I_n$. $X_{n \times (p+1)}$ also called as `Design Matrix`.

#### 4.2.1. OLS Estimation of Regression Parameters · 回归参数的最小二乘估计

$$
\begin{aligned}
&\begin{aligned}
    \mathcal{P}: \min{S(\beta)} & = \sum_{i=1}^{n}{\left\vert y_i - \hat{y_i} \right\vert ^2} = \sum_{i=1}^{n}{\left\vert y_i - \sum_{i=1}^{p}{x_{ij}\beta_j} \right\vert ^2} \\
    & = \Vert y - X\beta^T \Vert_2^2
\end{aligned} \\\\

& \begin{aligned}
\text{Let } \frac{d\Vert y - X\beta^T \Vert_2^2}{dX} = 2\Vert y - X\beta^T \Vert_2 & = 0 \\
y - X\beta^T & = 0 \\
X\beta^T & = y \\
X^TX\beta^T & = X^Ty \\
(X^TX)^{-1}X^TX\beta^T & = (X^TX)^{-1}X^Ty \\
\beta^T & = (X^TX)^{-1}X^Ty
\end{aligned}
\end{aligned}
\tag{4.12}\label{4.12}
$$

where $\Vert · \Vert_2$ is `L2 Norm` $=\sqrt{r^2_1+r^2_2+\cdots+r^2_n}$.

And we usually take the Var. of `Residual`
$$
\left\{\begin{aligned}
& \hat{\sigma}^2 = \frac{\hat{\epsilon}^T\hat{\epsilon}}{n-p-1} =\frac{SSE?}{n-p-1} \\\\
& \text{SD}(\hat{\beta_i}) = \hat{\sigma}\sqrt{c_{ii}}, \quad\quad i=0,1,\cdots,p
\end{aligned}\right.
\tag{4.13}\label{4.13}
$$
where $\hat{\epsilon} = y-X\hat{\beta}$, and $c_{ii}$ is the $i_{th}$ element of the diagonal of $C=(X^TX)^{-1}$.

#### 4.2.2. Significance Test of Regression Parameters · 回归参数的显著性检验

$\cases{H_0: \beta_0 = \beta_1 = \cdots = \beta_p = 0 \\ H_1: \beta_0, \beta_1, \cdots, \beta_p \text{ is not all 0}}$

1. **t Test**
   $$
   \begin{aligned}
   & T = \frac{\hat{\beta_1}}{\text{SD}\left( \hat{\beta_1} \right)} = \frac{\hat{\beta_1}}{\frac{\hat{\sigma}}{\sqrt{S_{xx}}}} = \frac{\hat{\beta_1}\sqrt{S_{xx}}}{\hat{\sigma}} \sim t(n-p-1) \\
   & W = \vert T \vert \geqslant t_{\frac{\alpha}{2}}(n-p-1)
   \end{aligned}
   \tag{4.14}\label{4.14}
   $$

2. **F Test**
   $$
   \begin{aligned}
   & F = \frac{\frac{SSR}{p}}{\frac{SSE}{n-p-1}} \sim F(1, n-p-1) \\
   & W = \vert F \vert \geqslant F_{\frac{\alpha}{2}}(1, n-p-1)
   \end{aligned}
   \tag{4.15}\label{4.15}
   $$

#### 4.1.3. Interval Estimation of Regression Parameters · 回归参数的区间估计

$$
\begin{aligned}
& T_i = \frac{\hat{\beta_i}-\beta_i}{\text{SD}\left( \hat{\beta_i} \right)} \sim t(n-p-1), \quad\quad i=0,1,\cdots,p \\
& P \left\{\left\vert \frac{\hat{\beta_i}-\beta_i}{\text{SD}\left( \hat{\beta_i} \right)} \right\vert \leqslant t_{\frac{\alpha}{2}}(n-p-1)\right\} = \alpha \\\\
& \left[ \hat{\beta_i} - \text{SD}\left( \hat{\beta_i} \right)t_{\frac{\alpha}{2}}(n-p-1), \hat{\beta_i} + \text{SD}\left( \hat{\beta_i} \right)t_{\frac{\alpha}{2}}(n-p-1) \right]
\end{aligned}
\tag{4.16}\label{4.16}
$$

