# MM

## 1. Обозначения и функция потерь

$$
x\in\mathcal X=\mathcal S^D,\quad \mathcal S=\{0,\ldots,S-1\},\quad
X_0=\{x_0^{(i)}\}_{i=1}^{N_0}\sim p_0,\quad
X_1=\{x_1^{(j)}\}_{j=1}^{N_1}\sim p_1.
$$

$$
\Delta_M=\{z\in\mathbb R_+^M:\mathbf1^\top z=1\},\qquad
\theta=\{\beta_k,r_k^d\}_{k,d},\quad \beta\in\Delta_K,\quad r_k^d\in\Delta_S.
$$

$$
q^{\mathrm{ref}}(x_1\mid x_0)=\prod_dQ_d[x_0^d,x_1^d],\qquad
Q_d[a,s]>0,\quad \sum_sQ_d[a,s]=1.
$$

Выборки независимы и фиксированы. $Q_d[a,s]=q^{\mathrm{ref}}(x_1^d=s\mid x_0^d=a)$ — заданная вероятность полного перехода от начала к концу по координате $d$. Начальные $\beta_k,r_k^d[s]$ строго положительны. Дополнительной регуляризации нет.

$$
v_\theta(x_1)=\sum_k\beta_k\prod_dr_k^d[x_1^d],
\qquad
c_\theta(x_0)=\sum_{x_1}q^{\mathrm{ref}}(x_1\mid x_0)v_\theta(x_1),
$$

$$
q_\theta(x_1\mid x_0)
=\frac{q^{\mathrm{ref}}(x_1\mid x_0)v_\theta(x_1)}{c_\theta(x_0)}.
$$

Совместное распределение модели:

$$
\boxed{q_\theta(x_0,x_1)=p_0(x_0)q_\theta(x_1\mid x_0),\qquad q_\theta(x_0)=p_0(x_0).}
$$

$v_\theta$ — потенциал. Конечный маргинал модели равен $q_\theta(x_1)=\sum_{x_0}q_\theta(x_0,x_1)$.

Для точек двух выборок обозначим

$$
c_i(\theta):=c_\theta(x_0^{(i)}),\qquad
v_j(\theta):=v_\theta(x_1^{(j)}).
$$


Раскроем $c_i$, подставив факторизованный переход и сумму компонент потенциала:

$$
\begin{aligned}
c_i(\theta)
&=\sum_{x_1\in\mathcal X}
q^{\mathrm{ref}}(x_1\mid x_0^{(i)})v_\theta(x_1)\\
&=\sum_{x_1\in\mathcal X}
\left(\prod_{d=1}^D Q_d[x_0^{(i),d},x_1^d]\right)
\left(\sum_{k=1}^K\beta_k\prod_{d=1}^D r_k^d[x_1^d]\right)\\
&=\sum_{k=1}^K\beta_k
\sum_{x_1\in\mathcal X}
\prod_{d=1}^D\left(Q_d[x_0^{(i),d},x_1^d]r_k^d[x_1^d]\right).
\end{aligned}
$$

<!-- Все суммы конечны, поэтому суммы по $x_1$ и $k$ можно переставить. Вес $\beta_k$ не зависит от $x_1$ и выносится за внутреннюю сумму; множители перехода и потенциала объединяются по координате $d$. -->

Поскольку $\mathcal X=\mathcal S^D$, сумма по вектору $x_1$ — это сумма по всем сочетаниям его координат. Каждый множитель зависит только от своей координаты, поэтому по распределительному закону

$$
\begin{aligned}
&\sum_{x_1\in\mathcal X}
\prod_{d=1}^D\left(Q_d[x_0^{(i),d},x_1^d]r_k^d[x_1^d]\right)\\
&\quad=\sum_{s_1=0}^{S-1}\cdots\sum_{s_D=0}^{S-1}
\prod_{d=1}^D\left(Q_d[x_0^{(i),d},s_d]r_k^d[s_d]\right)\\
&\quad=\prod_{d=1}^D
\left(\sum_{s=0}^{S-1}Q_d[x_0^{(i),d},s]r_k^d[s]\right).
\end{aligned}
$$

Введём сокращения для вероятности перехода и суммы по одной координате:

$$
Q_{ids}:=Q_d[x_0^{(i),d},s],\qquad
u_{ikd}:=\sum_{s=0}^{S-1}Q_{ids}r_k^d[s].
$$

Тогда

$$
\boxed{c_i(\theta)=\sum_{k=1}^K\beta_k\prod_{d=1}^D u_{ikd}.}
$$

Для $v_j$ получаем

$$
a_{jk}(\theta):=\beta_k\prod_{d=1}^D r_k^d[x_1^{(j),d}],\qquad
\boxed{v_j(\theta)=\sum_{k=1}^K a_{jk}(\theta).}
$$

$$
\mathcal L(\theta)=\mathbb E_{p_0}\log c_\theta(x_0)-\mathbb E_{p_1}\log v_\theta(x_1),
\qquad
\mathrm{KL}(q^*\|q_\theta)=\mathcal L(\theta)-\mathcal L^*,\quad \partial_\theta\mathcal L^*=0.
$$

$$
\begin{aligned}
\mathrm{KL}(q^*\|q_\theta)
&=\sum_{x_0,x_1}q^*(x_0,x_1)\log\frac{q^*(x_0,x_1)}{q_\theta(x_0,x_1)}\\
&=\mathbb E_{x_0\sim p_0}\mathrm{KL}\!\left(q^*(\cdot\mid x_0)\,\middle\|\,q_\theta(\cdot\mid x_0)\right).
\end{aligned}
$$

Здесь $q^*(x_0,x_1)$ — оптимальное сопряжение с маргиналами $p_0,p_1$ для референса $q^{\mathrm{ref}}(x_0,x_1)=p_0(x_0)q^{\mathrm{ref}}(x_1\mid x_0)$. Его условное распределение и значение $\mathcal L^*$:

$$
q^*(x_1\mid x_0)=\frac{v^*(x_1)q^{\mathrm{ref}}(x_1\mid x_0)}{c^*(x_0)},\qquad
\mathcal L^*=\mathbb E_{p_0}\log c^*(x_0)-\mathbb E_{p_1}\log v^*(x_1).
$$

$\mathcal L^*$ — значение цели для точного SB; оно не обязано достигаться при конечном $K$.

$$
\boxed{
\min_\theta\widehat{\mathcal L}(\theta),\qquad
\widehat{\mathcal L}(\theta)=
\underbrace{\frac1{N_0}\sum_i\log c_i(\theta)}_{\widehat{\mathcal L}_+}
+\underbrace{\left(-\frac1{N_1}\sum_j\log v_j(\theta)\right)}_{\widehat{\mathcal L}_-}.
}
$$

Для вычисления $\widehat{\mathcal L}$ достаточно двух выборок; знание $q^*$ не требуется. Верхний индекс $(t)$ обозначает значение в начале внешней MM-итерации.

## 2. Верхняя оценка первого слагаемого

Представить через log-sum-exp:

$$
\eta_{ik}(\theta)=\log\beta_k+\sum_d\log u_{ikd},\qquad
\log c_i(\theta)=\operatorname{LSE}(\eta_i(\theta)),\qquad
\operatorname{LSE}(z)=\log\sum_{k=1}^K e^{z_k}.
$$

**Касательные к внутренним логарифмам:**

$$
\log z\le\log z^{(t)}+\frac{z-z^{(t)}}{z^{(t)}},\qquad z,z^{(t)}>0,
$$

$$
\log\beta_k\le\log\beta_k^{(t)}+
\frac{\beta_k-\beta_k^{(t)}}{\beta_k^{(t)}},\qquad
\log u_{ikd}\le\log u_{ikd}^{(t)}+
\frac{u_{ikd}-u_{ikd}^{(t)}}{u_{ikd}^{(t)}}.
$$

$$
\boxed{
\begin{aligned}
\delta_{ik}(\theta)
&=\frac{\beta_k-\beta_k^{(t)}}{\beta_k^{(t)}}
+\sum_d\frac{u_{ikd}-u_{ikd}^{(t)}}{u_{ikd}^{(t)}}\\
&=\frac{\beta_k-\beta_k^{(t)}}{\beta_k^{(t)}}
+\sum_{d,s}\frac{Q_{ids}}{u_{ikd}^{(t)}}
\bigl(r_k^d[s]-r_k^{d,(t)}[s]\bigr).
\end{aligned}
}
$$

$$
\eta_i(\theta)\le\widetilde\eta_i(\theta):=\eta_i^{(t)}+\delta_i(\theta),\qquad
\delta_i(\theta^{(t)})=0.
$$

Покомпонентная монотонность LSE:

$$
\partial_{z_k}\operatorname{LSE}(z)=\operatorname{softmax}_k(z)>0
\quad\Longrightarrow\quad
\log c_i(\theta)\le\operatorname{LSE}(\widetilde\eta_i(\theta)).
$$

**[Оценка Бёнинга (Böhning bound)](https://www.ism.ac.jp/editsec/aism/pdf/044_1_0197.pdf):**
<!-- 
Для $M$ свободных аргументов и одной базовой компоненты с нулевым аргументом:

$$
\operatorname{lse}(\eta)=\log\left(1+\sum_{k=1}^{M}e^{\eta_k}\right),\qquad
S_k(\psi)=\frac{e^{\psi_k}}{1+\sum_{h=1}^{M}e^{\psi_h}},\qquad
\eta,\psi\in\mathbb R^M.
$$

$$
\boxed{
\operatorname{lse}(\eta)\le\frac12\eta^\top A\eta-b_\psi^\top\eta+c_\psi,
}
\qquad
A=\frac12\left(I_M-\frac{\mathbf1_M\mathbf1_M^\top}{M+1}\right),
$$

$$
b_\psi=A\psi-S(\psi),\qquad
c_\psi=\frac12\psi^\top A\psi-S(\psi)^\top\psi+\operatorname{lse}(\psi).
$$ -->

<!--  Для полной LSE с $K$ аргументами используем симметричную форму: -->

$$
B=\frac12\left(I_K-\frac1K\mathbf1\mathbf1^\top\right),\qquad
b_\psi=B\psi-\operatorname{softmax}(\psi),\qquad
c_\psi=\frac12\psi^\top B\psi-\operatorname{softmax}(\psi)^\top\psi
+\operatorname{LSE}(\psi),
$$

$$
\boxed{
\begin{aligned}
\operatorname{LSE}(z)
&\le\frac12z^\top Bz-(b_\psi)^\top z+c_\psi\\
&=\operatorname{LSE}(\psi)+\operatorname{softmax}(\psi)^\top(z-\psi)
+\frac12(z-\psi)^\top B(z-\psi),\qquad z,\psi\in\mathbb R^K.
\end{aligned}
}
$$

Равенство достигается при $\eta=\psi$.

Зафиксировать нормированные вклады компонент в $c_i^{(t)}$:

$$
\boxed{
\pi_{ik}^{(t)}=\operatorname{softmax}_k(\eta_i^{(t)})
=\frac{\beta_k^{(t)}\prod_du_{ikd}^{(t)}}{c_i^{(t)}},\qquad
\sum_k\pi_{ik}^{(t)}=1.
}
$$

Подставить $z=\widetilde\eta_i(\theta)$, $\psi=\eta_i^{(t)}$:

$$
\log c_i(\theta)
\le\operatorname{LSE}(\widetilde\eta_i(\theta))
\le\log c_i^{(t)}+(\pi_i^{(t)})^\top\delta_i(\theta)
+\frac12\delta_i(\theta)^\top B\delta_i(\theta).
$$

<small>Верхние оценки аргументов подставляются сначала в монотонную LSE. Квадратичная граница сама не обязана быть покомпонентно монотонной.</small>

$$
\boxed{
G_+(\theta\mid\theta^{(t)})
=\frac1{N_0}\sum_i\left[
\log c_i^{(t)}+\sum_k\pi_{ik}^{(t)}\delta_{ik}(\theta)
+\frac14\left(\sum_k\delta_{ik}(\theta)^2
-\frac1K\left[\sum_k\delta_{ik}(\theta)\right]^2\right)
\right].
}
$$

<!-- $$
\widehat{\mathcal L}_+(\theta)\le G_+(\theta\mid\theta^{(t)}),\qquad
G_+(\theta^{(t)}\mid\theta^{(t)})=\widehat{\mathcal L}_+(\theta^{(t)}),
$$

$$
\left.\nabla_\theta G_+(\theta\mid\theta^{(t)})\right|_{\theta=\theta^{(t)}}
=\nabla\widehat{\mathcal L}_+(\theta^{(t)}).
$$ -->

## 3. Верхняя оценка второго слагаемого через ELBO

Ввести распределение по компонентам $\gamma_j\in\Delta_K$. Для положительных аргументов:

$$
\log v_j(\theta)
=\log\sum_k\gamma_{jk}\frac{a_{jk}(\theta)}{\gamma_{jk}}
\ge\sum_k\gamma_{jk}\log\frac{a_{jk}(\theta)}{\gamma_{jk}}
=:\mathcal E_j(\theta,\gamma_j).
$$

$$
\mathcal E_j(\theta,\gamma_j)
=\sum_k\gamma_{jk}\left[\log\beta_k
+\sum_d\log r_k^d[x_1^{(j),d}]-\log\gamma_{jk}\right].
$$

$\mathcal E_j$ — ELBO для скрытого индекса компоненты $k$; отрицательная ELBO даёт верхнюю границу:

$$
\boxed{
\widehat{\mathcal L}_-(\theta)\le G_-(\theta,\gamma)
:=-\frac1{N_1}\sum_j\mathcal E_j(\theta,\gamma_j).
}
$$

Найти $\gamma$ при фиксированной $\theta^{(t)}$:

$$
\mathcal J_j(\gamma_j,\nu_j)
=\sum_k\gamma_{jk}(\log\gamma_{jk}-\log a_{jk}^{(t)})
+\nu_j\left(\sum_k\gamma_{jk}-1\right),
$$

$$
\partial_{\gamma_{jk}}\mathcal J_j
=\log\gamma_{jk}+1-\log a_{jk}^{(t)}+\nu_j=0,
\qquad \nabla_{\gamma_j}^2\mathcal J_j=\operatorname{diag}(1/\gamma_{jk})\succ0,
$$

$$
\boxed{
\gamma_{jk}^{(t)}=\frac{a_{jk}^{(t)}}{\sum_ha_{jh}^{(t)}}
=\frac{\beta_k^{(t)}\prod_dr_k^{d,(t)}[x_1^{(j),d}]}{v_j^{(t)}}.
}
$$

Разрыв и касание:

$$
\log v_j(\theta)-\mathcal E_j(\theta,\gamma_j)
=\mathrm{KL}\!\left(\gamma_j\,\middle\|\,\frac{a_j(\theta)}{v_j(\theta)}\right),
$$

$$
G_-(\theta,\gamma)-\widehat{\mathcal L}_-(\theta)
=\frac1{N_1}\sum_j\mathrm{KL}\!\left(\gamma_j\,\middle\|\,\frac{a_j(\theta)}{v_j(\theta)}\right)\ge0,
$$

$$
G_-(\theta^{(t)},\gamma^{(t)})=\widehat{\mathcal L}_-(\theta^{(t)}).
$$

<small>На границе используется 0·log 0 = 0; положительный вес при нулевой вероятности даёт +∞ в отрицательной ELBO. Формула разрыва применяется при vⱼ > 0.</small>

## 4. Полная мажоранта

Накопить и зафиксировать:

$$
n_k^{(t)}=\frac1{N_1}\sum_j\gamma_{jk}^{(t)},\qquad
m_{kds}^{(t)}=\frac1{N_1}\sum_j\gamma_{jk}^{(t)}\mathbf1\{x_1^{(j),d}=s\},
$$

$$
\sum_kn_k^{(t)}=1,\qquad \sum_sm_{kds}^{(t)}=n_k^{(t)}.
$$

$$
\boxed{
\begin{aligned}
G_t(\theta)&:=G_+(\theta\mid\theta^{(t)})+G_-(\theta,\gamma^{(t)})\\
&=\frac1{N_0}\sum_i\left[(\pi_i^{(t)})^\top\delta_i(\theta)
+\frac12\delta_i(\theta)^\top B\delta_i(\theta)\right]\\
&\quad-\sum_kn_k^{(t)}\log\beta_k
-\sum_{k,d,s}m_{kds}^{(t)}\log r_k^d[s]+C_t,
\end{aligned}
}
$$

$$
C_t=\frac1{N_0}\sum_i\log c_i^{(t)}
+\frac1{N_1}\sum_{j,k}\gamma_{jk}^{(t)}\log\gamma_{jk}^{(t)}.
$$

$$
\boxed{
\widehat{\mathcal L}(\theta)\le G_t(\theta),\qquad
G_t(\theta^{(t)})=\widehat{\mathcal L}(\theta^{(t)}),\qquad
\nabla G_t(\theta^{(t)})=\nabla\widehat{\mathcal L}(\theta^{(t)}).
}
$$

## 5. Совместная задача и условия оптимальности

Собрать параметры в вектор $\theta\in\mathbb R^P$, $P=K+KDS$. Индекс $a$ нумерует его координаты. Матрица $E$ суммирует каждый блок: $\beta$ и все $r_k^d$.

$$
\mathcal D=\{\theta\ge0:E\theta=\mathbf1\}
=\Delta_K\times\prod_{k,d}\Delta_S.
$$

$$
\delta_i(\theta)=A_i(\theta-\theta^{(t)}),\qquad
(A_i)_{k,\beta_h}=\frac{\mathbf1\{k=h\}}{\beta_k^{(t)}},\qquad
(A_i)_{k,r_h^d[s]}=\mathbf1\{k=h\}\frac{Q_{ids}}{u_{ikd}^{(t)}}.
$$

$$
H_t=\frac1{N_0}\sum_iA_i^\top BA_i,\qquad
g_t=\frac1{N_0}\sum_iA_i^\top\pi_i^{(t)},\qquad
b_{t,\beta_k}=n_k^{(t)},\quad b_{t,r_k^d[s]}=m_{kds}^{(t)}.
$$

$$
\boxed{
\begin{aligned}
\theta_t^+&\in\arg\min_{\theta\in\mathcal D}F_t(\theta),\\
F_t(\theta)&:=G_t(\theta)-C_t\\
&=\frac12(\theta-\theta^{(t)})^\top H_t(\theta-\theta^{(t)})
+g_t^\top(\theta-\theta^{(t)})-\sum_ab_{t,a}\log\theta_a.
\end{aligned}
}
$$

Соглашение на границе:

$$
-b\log z:=
\begin{cases}
-b\log z,&b>0,\ z>0,\\
+\infty,&b>0,\ z=0,\\
0,&b=0,\ z\ge0.
\end{cases}
$$

**Выпуклость и существование:**

$$
h^\top H_th=\frac1{N_0}\sum_i(A_ih)^\top B(A_ih)\ge0,
$$

$$
\nabla F_t=H_t(\theta-\theta^{(t)})+g_t-b_t\oslash\theta,\qquad
\nabla^2F_t=H_t+\operatorname{diag}(b_{t,a}/\theta_a^2)\succeq0.
$$

$$
\mathcal D\text{ компактна},\quad F_t\text{ полунепрерывна снизу},\quad
F_t(\theta^{(t)})<\infty
\quad\Longrightarrow\quad \arg\min_{\mathcal D}F_t\ne\varnothing.
$$

$$
\forall a:\ b_{t,a}>0
\quad\Longrightarrow\quad
\nabla^2F_t\succ0,\quad \theta_t^+>0,\quad
\text{минимум единственен}.
$$

<small>При нулевых счётчиках минимум может быть граничным и неединственным. Деление ⊘ покоординатное; при bₜ,ₐ = 0 вклад логарифма в градиент и гессиан равен нулю, включая границу.</small>

**KKT:**

$$
\mathcal J_t=F_t(\theta)+\lambda^\top(E\theta-\mathbf1)-\mu^\top\theta,
$$

$$
\boxed{
\begin{aligned}
&\nabla F_t(\theta_t^+)+E^\top\lambda-\mu=0,\\
&E\theta_t^+=\mathbf1,\quad \theta_t^+\ge0,\quad
\mu\ge0,\quad \mu_a\theta_{t,a}^+=0.
\end{aligned}
}
$$

Положительная допустимая точка даёт условие Слейтера, поэтому KKT необходимы. Достаточность для любой допустимой $\theta$:

$$
\begin{aligned}
F_t(\theta)-F_t(\theta_t^+)
&\ge\nabla F_t(\theta_t^+)^\top(\theta-\theta_t^+)\\
&=(-E^\top\lambda+\mu)^\top(\theta-\theta_t^+)
=\mu^\top\theta\ge0.
\end{aligned}
$$

$$
\boxed{\text{KKT}\quad\Longleftrightarrow\quad
\theta_t^+\text{ — глобальный минимум мажоранты}.}
$$

## 6. Поиск параметров M-шага

**Производные по весам и ядрам:**

$$
\psi_{ik}(\theta)=\pi_{ik}^{(t)}+(B\delta_i(\theta))_k,
$$

$$
\frac{\partial F_t}{\partial\beta_k}
=\frac1{N_0\beta_k^{(t)}}\sum_i\psi_{ik}(\theta)-\frac{n_k^{(t)}}{\beta_k},
$$

$$
\frac{\partial F_t}{\partial r_k^d[s]}
=\frac1{N_0}\sum_i\frac{Q_{ids}}{u_{ikd}^{(t)}}\psi_{ik}(\theta)
-\frac{m_{kds}^{(t)}}{r_k^d[s]}.
$$

Во внутреннем минимуме:

$$
\frac{\partial F_t}{\partial\beta_k}+\lambda_\beta=0,\qquad
\frac{\partial F_t}{\partial r_k^d[s]}+\lambda_{kd}=0,\qquad
\sum_k\beta_k=1,\quad\sum_sr_k^d[s]=1.
$$

<small>Уравнения связаны через ψ(θ) и решаются совместно.</small>

**Метод Ньютона при положительных счётчиках:**

Внутренняя итерация $\ell$; начать с $\vartheta^{(0)}=\theta^{(t)}$.

$$
f_\ell=\nabla F_t(\vartheta^{(\ell)}),\qquad
W_\ell=H_t+\operatorname{diag}\left(b_{t,a}/(\vartheta_a^{(\ell)})^2\right)\succ0.
$$

$$
d_\ell=\arg\min_{Ed=0}\left(f_\ell^\top d+\frac12d^\top W_\ell d\right),
\qquad
\boxed{
\begin{pmatrix}W_\ell&E^\top\\E&0\end{pmatrix}
\begin{pmatrix}d_\ell\\\nu_\ell\end{pmatrix}
=\begin{pmatrix}-f_\ell\\0\end{pmatrix}.
}
$$

$$
Ed_\ell=0,\qquad
f_\ell^\top d_\ell=-d_\ell^\top W_\ell d_\ell<0\quad(d_\ell\ne0).
$$

Выбрать $0<\rho,\xi<1$, $0<\sigma<1/2$. Начальный шаг и уменьшение:

$$
\alpha_{\max}=\min_{a:d_{\ell,a}<0}
\frac{-\vartheta_a^{(\ell)}}{d_{\ell,a}},\qquad
\alpha_\ell=\min\{1,\rho\alpha_{\max}\},\qquad
\alpha_\ell\leftarrow\xi\alpha_\ell,
$$

пока не выполнено условие Армихо:

$$
F_t(\vartheta^{(\ell)}+\alpha_\ell d_\ell)
\le F_t(\vartheta^{(\ell)})+\sigma\alpha_\ell f_\ell^\top d_\ell.
$$

$$
\boxed{\vartheta^{(\ell+1)}=\vartheta^{(\ell)}+\alpha_\ell d_\ell,\qquad
E\vartheta^{(\ell+1)}=\mathbf1,\quad\vartheta^{(\ell+1)}>0.}
$$

<small>Минимум по пустому множеству в αmax равен +∞. При d = 0 выполнены условия внутреннего оптимума.</small>

$$
b_t>0\ \Longrightarrow\
\{\theta\in\mathcal D:F_t(\theta)\le F_t(\theta^{(t)})\}
\subset\{\theta:\theta_a\ge\epsilon_t>0\},
$$

$$
0<m_tI\preceq\nabla^2F_t\preceq M_tI
\quad\Longrightarrow\quad
\vartheta^{(\ell)}\longrightarrow\theta_t^+
\quad\text{для Ньютона с поиском шага Армихо}.
$$

**Нулевые счётчики: барьерный поиск с $\tau\downarrow0$:**

$$
F_{t,\tau}(\theta)=F_t(\theta)-\tau\sum_{a=1}^P\log\theta_a,\qquad
\theta_{t,\tau}=\arg\min_{E\theta=\mathbf1,\ \theta>0}F_{t,\tau}(\theta),\quad\tau>0.
$$

Применить те же шаги Ньютона с $b_t+\tau\mathbf1$ вместо $b_t$. Для точного решения барьерной задачи:

$$
\nabla F_t(\theta_{t,\tau})+E^\top\lambda-\mu=0,\qquad
\mu_a=\frac{\tau}{\theta_{t,\tau,a}}>0,\qquad
\mu^\top\theta_{t,\tau}=P\tau,
$$

$$
\boxed{
0\le F_t(\theta_{t,\tau})-F_t(\theta_t^+)
\le\nabla F_t(\theta_{t,\tau})^\top(\theta_{t,\tau}-\theta_t^+)
=P\tau-\mu^\top\theta_t^+\le P\tau.
}
$$

<small>Барьер используется для приближённого решения исходной задачи; τ уменьшается, например τ ← ξτ. Все предельные точки точных барьерных решений оптимальны для Fₜ.</small>

**Сертификат точности для текущей допустимой точки:**

Пусть $f=\nabla F_t(\vartheta)$, а $\mathcal B$ пробегает блоки $\beta,r_k^d$.

$$
\begin{aligned}
0\le F_t(\vartheta)-F_t(\theta_t^+)
&\le f^\top(\vartheta-\theta_t^+)\\
&\le f^\top\vartheta-\min_{z\in\mathcal D}f^\top z
=f^\top\vartheta-\sum_{\mathcal B}\min_{a\in\mathcal B}f_a
=:\mathfrak g_t(\vartheta).
\end{aligned}
$$

$$
\boxed{\mathfrak g_t(\vartheta)\le\varepsilon_{\mathrm{opt}}
\quad\Longrightarrow\quad
F_t(\vartheta)-\min_{\mathcal D}F_t\le\varepsilon_{\mathrm{opt}}.}
$$

## 7. Итерация и монотонность

$$
\theta^{(t)}\longrightarrow
\bigl(u^{(t)},c^{(t)},\pi^{(t)},\gamma^{(t)},n^{(t)},m^{(t)}\bigr)
\longrightarrow (H_t,g_t,b_t,C_t)\longrightarrow F_t.
$$

Решить M-задачу при фиксированных коэффициентах. Принять положительную допустимую точку $\theta^{(t+1)}$ с

$$
F_t(\theta^{(t+1)})\le F_t(\theta^{(t)}),\qquad
\mathfrak g_t(\theta^{(t+1)})\le\varepsilon_{\mathrm{opt}}.
$$

$$
\boxed{
\widehat{\mathcal L}(\theta^{(t+1)})
\le G_t(\theta^{(t+1)})
\le G_t(\theta^{(t)})
=\widehat{\mathcal L}(\theta^{(t)}).
}
$$

<small>Для невозрастания достаточно первого условия; второе контролирует точность M-шага. Уменьшение барьерной функции само по себе не заменяет проверку Fₜ. Положительные принятые точки позволяют заново строить касательные на следующей итерации.</small>

Остановка внешних итераций:

$$
0\le\widehat{\mathcal L}(\theta^{(t)})-\widehat{\mathcal L}(\theta^{(t+1)})
\le\varepsilon\max\{1,|\widehat{\mathcal L}(\theta^{(t)})|\}.
$$

<small>Гарантия относится к фиксированной выборке. Глобальная оптимальность M-шага не означает глобальной оптимальности исходного loss; малое изменение loss — критерий остановки, а не сертификат его глобального минимума.</small>

## 8. Эквивалентная нормировка параметров

Для положительных ненормированных весов $\widetilde\beta_k,\widetilde r_k^d[s]$:

$$
Z_{dk}=\sum_s\widetilde r_k^d[s],\qquad
r_k^d[s]=\frac{\widetilde r_k^d[s]}{Z_{dk}},\qquad
w_k=\widetilde\beta_k\prod_dZ_{dk},\quad W=\sum_kw_k,\qquad
\beta_k=\frac{w_k}{W}.
$$

$$
v_\theta=\frac{\widetilde v}{W},\qquad c_\theta=\frac{\widetilde c}{W}
\quad\Longrightarrow\quad
q_\theta(x_1\mid x_0)=q_{\widetilde\theta}(x_1\mid x_0),\quad
q_\theta(x_0,x_1)=q_{\widetilde\theta}(x_0,x_1),
$$

$$
\widehat{\mathcal L}(\theta)
=\frac1{N_0}\sum_i(\log\widetilde c_i-\log W)
-\frac1{N_1}\sum_j(\log\widetilde v_j-\log W)
=\widehat{\mathcal L}(\widetilde\theta).
$$

В лог-параметрах $\ell_{dsk}=\log\widetilde r_k^d[s]$, $\alpha_k=\log\widetilde\beta_k$:

$$
z_{dk}=\operatorname{LSE}_s\ell_{dsk},\qquad
\log r_k^d[s]=\ell_{dsk}-z_{dk},
$$

$$
\widetilde\alpha_k=\alpha_k+\sum_dz_{dk},\qquad
\log\beta_k=\widetilde\alpha_k-\operatorname{LSE}_h\widetilde\alpha_h.
$$

<small>Нормировка выполняется до построения мажоранты. Hₜ, gₜ, bₜ и δᵢ вычисляются в координатах β, r на симплексах; выпуклость M-задачи относится к этим координатам.</small>
