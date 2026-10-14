# Phase object approximation

The phase object approximation assumes the specimen is very thin and
composed of light atoms. The effect of the specimen will then only
produce a spatially varying phase shift in the electron wave function,
depending on the electrostatic (scalar) and magnetic (vector) potential
landscape of the sample [@yu2023novel]:
$$\varphi(\Vec{r})=\frac{\pi}{\lambda E}\int_{-\infty}^{+\infty}V(\Vec{r}, z)dz-\frac{e}{\hbar}\int_{-\infty}^{+\infty}A_z(\Vec{r}, z)dz$$
Here, $\Vec{r}$ represents the $(x, y)$ coordinate of the electron. If
we call the incoming wavefunction $\psi_{inc}$ and the wave function
passing through the specimen $\psi_t$, we get: $$\label{eq:phase_object}
    \psi_t(\Vec{r})=\exp{i\varphi(\Vec{r})}\psi_{inc}(\Vec{r})$$ Note
that the intensity of the outgoing wavefunction is the same as the
incoming wavefunction. A CCD detector cannot distinguish between the two
wavefunctions, as it only measures the intensity. An additional
approximation is that the phase shift is very small, allowing for the
expansion of the exponential term: $$\label{eq:weak_phase_object}
    \psi_t(\Vec{r})=(1+i\varphi(\Vec{r}))\psi_{inc}(\Vec{r})$$

# Simplified through focal series reconstruction

I will first try to give a very simplified description of the through
focal series reconstruction method based on [@kirkland2010advanced]. For
this, I will only take defocus and third-order spherical aberration into
account, while neglecting all other aberrations. Later, I will try to
give a more realistic problem formulation.\
I will model the electron microscope by a single objective lens, as
shown on Figure [1](#fig:simplified_TEM){reference-type="ref"
reference="fig:simplified_TEM"}. We can assume that the condenser lens
only influences the properties of the rays illuminating the sample, and
the projector lens only magnifies the image produces by the objective
lens. The specimen is illuminated with a broad, parallel beam, as is
standard in conventional transmission electron microscopy (CTEM).

![Simplified model of a CTEM, modeled by a single objective lens.
[@kirkland2010advanced]](simplified_TEM.png){#fig:simplified_TEM
width="50%"}

If a plane wave falls onto the objective lens, ideally it would produce
a spherical wave converging to a point on the optical axis of the lens.
However, the true wavefronts will differ from a perfect spherical
surface due to aberrations (in our case defocus and spherical
aberration), as shown on Figure
[2](#fig:aberrations_electron_wavefronts){reference-type="ref"
reference="fig:aberrations_electron_wavefronts"}[^1]. Due to the
aberrations, there is a *deviation* $\delta$ between a perfect spherical
wave and the true wave. This distance depends on the angle between the
electron trajectory and the optical axis, denoted by $\alpha$. As the
aberrations are worse at larger angles, $\delta(\alpha)$ is larger for
larger $\alpha$. This deviation corresponds to a *phase error*:
$$\chi(\alpha)=\frac{2\pi}{\lambda}\delta(\alpha)$$

![Effect of spherical aberration on electron trajectories and
wavefronts. [@kirkland2010advanced], modified
figure.](aberrations_electron_wavefronts.png){#fig:aberrations_electron_wavefronts
width="80%"}

If we assume the lens is perfectly symmetric (neglecting coma and
astigmatism), the deviation $\delta(\alpha)$ can only depend on even
powers of $\alpha=\sqrt{\alpha_x^2+\alpha_y^2}$:
$$\chi(\alpha)=\frac{2\pi}{\lambda}\delta(\alpha)=\frac{2\pi}{\lambda}\left(\frac{1}{2}C_1\alpha^2+\frac{1}{4}C_3\alpha^4+\mathcal{O}(\alpha^6)\right)$$
We will now neglect higher order terms. The defocus and third-order
spherical aberration are now defined as: $$\begin{aligned}
        \Delta f&=C_1\\
        C_{s3}&=C_3
    \end{aligned}$$ The phase error can then be described as:
$$\chi(\alpha)=\frac{2\pi}{\lambda}\left(\frac{1}{4}C_{s3}\alpha^4+\frac{1}{2}\Delta f\alpha^2\right)$$
Or in terms of the spatial frequency $\Vec{k}$:
$$\label{eq:phase_error_simplified}
    \begin{aligned}
        \alpha=\lambda k\Leftrightarrow\chi(k)&=\frac{2\pi}{\lambda}\left(\frac{1}{4}C_{s3}\lambda^4 k^4+\frac{1}{2}\Delta f\lambda^2 k^2\right)\\
        &=\pi\lambda k^2\left(\frac{1}{2}C_{s3}\lambda^2k^2+\Delta f\right)
    \end{aligned}$$ The defocus depends on the strength of the objective
lens, while the spherical aberration can be assumed to be constant for a
given specimen holder and beam energy.\
If we now consider an image that will be formed by the objective lens,
the phase error will shift the phase of each spatial frequency in the
image by a different amount. This is described by taking the Fourier
transform of the incident wavefunction $\psi_t$:
$$\FT{\psi_t(\Vec{r})}=\Psi_t(\Vec{k})$$ The wavefunction in the
backfocal plane $\Psi_i(\Vec{k})$ is affected by the phase error:
$$\Psi_i(\Vec{k})=\Psi_t(\Vec{k})\exp{-i\chi(\Vec{k})}=\Psi_t(\Vec{k})H_0(\Vec{k})$$
$H_0(\Vec{k})$ is the coherent phase transfer function and contributes
to the contrast transfer function of the microscope. In the image plane
the wavefunction $\psi_i$ will be the inverse Fourier transform of
$\Psi_i$:
$$\psi_i(\Vec{r})=\IFT{\Psi_i(\Vec{k})}=\IFT{\Psi_t(\Vec{k})H_0(\Vec{k})}=\psi_t(\Vec{r})\ast h_0(\Vec{r})$$
Where $h_0(\Vec{r})$ is the inverse Fourier transform of
$\exp{-i\chi(\Vec{k})}$. Since the detector can only measure the
wavefunction intensity, the resulting image $g(\Vec{r})$ becomes:
$$g(\Vec{r})=\abs{\psi_t(\Vec{r})\ast h_0(\Vec{r})}^2$$ If we now
consider a phase object, we can use equation
[\[eq:phase_object\]](#eq:phase_object){reference-type="ref"
reference="eq:phase_object"} to rewrite the resulting image:
$$g(\Vec{r})=\abs{\left[\exp{i\Delta\varphi(\Vec{r})}\psi_{inc}(\Vec{r})\right]\ast h_0(\Vec{r})}^2$$
If we consider a weak phase object and assume the incident wavefunction
is a plane wave ($\psi_{inc}=1$), this expression can be simplified
further:
$$\psi_i(\Vec{r})=\left[1+i\varphi(\Vec{r})\right]\ast h_0(\Vec{r})=1+i\left[\varphi(\Vec{r})\ast h_0(\Vec{r})\right]$$
$$\label{eq:weak_phase_object_image}
    \begin{aligned}
        g(\Vec{r})&=\abs{\psi_i(\Vec{r})}^2=\abs{1+i\left[\varphi(\Vec{r})\ast h_0(\Vec{r})\right]}^2\\
        &\approx 1+\varphi(\Vec{r})\ast\left[ih_0(\Vec{r})-ih_0^*(\Vec{r})\right]\\
        &= 1+\varphi(\Vec{r})\ast \left[-2\operatorname{Im}\left(h_0(\Vec{r})\right)\right]\\
        &= 1+2\varphi(\Vec{r})\ast h_{WP}(\Vec{r})
    \end{aligned}$$ $h_{WP}$ is defined as the the point spread
function. Its Fourier transform $H_{WP}$ is called the phase-contrast
transfer function and is related to the coherent phase transfer function
$H_0$ and the phase error $\chi(\Vec{k})$ as follows: $$\begin{aligned}
        H_{WP}(\Vec{k})&=\FT{h_{WP}(\Vec{r})}\\
        &=\frac{i}{2}\FT{h_0(\Vec{r})-h_0^*(\Vec{r})}\\
        &=\frac{i}{2}\left[H_0(\Vec{k})-H_0^*(\Vec{k})\right]\\
        &=\frac{i}{2}\left[\exp{-i\chi(\Vec{k})}-\exp{i\chi(\Vec{k})}\right]\\
        &=\sin{\chi(\Vec{k})}
    \end{aligned}$$ If we Fourier transform equation
[\[eq:weak_phase_object_image\]](#eq:weak_phase_object_image){reference-type="ref"
reference="eq:weak_phase_object_image"}, we can use the Fourier
convolution theorem to convert the convolution into a product:
$$G(\Vec{k})=\delta(\Vec{k})+2\Phi(\Vec{k})\sin{\chi(\Vec{k})}$$ Where
$\Phi(\Vec{k})$ is the Fourier transform of the phase shift
$\varphi(\Vec{r})$ and $G(\Vec{k})$ is the Fourier transform of the
image $g(\Vec{r})$. If we fill in equation
[\[eq:phase_error_simplified\]](#eq:phase_error_simplified){reference-type="ref"
reference="eq:phase_error_simplified"}, the equation becomes:
$$\label{eq:simplified_result_focal_series_reconstruction}
    G(\Vec{k})=\delta(\Vec{k})+2\Phi(\Vec{k})\sin{\left[\pi\lambda k^2\left(\frac{1}{2}C_{s3}\lambda^2k^2+\Delta f\right)\right]}$$
The goal of through focal series reconstruction is to take images with
varying defocus $\Delta f$ to try to reconstruct $\Phi$, which can be
Fourier transformed to the phase difference $\varphi$.\
This last step assumes the phase object approximation (POA) or the weak
phase object approximation (WPOA). If we do not make these
approximations, the calculation becomes a lot more complex as will be
demonstrated below.

# Linear Image Reconstruction

I will now try to give a more thorough description of traditional
reconstruction methods, based on [@CarterWilliams2016TEM].\
The book makes a distinction between linear reconstruction and nonlinear
reconstruction. It is claimed that only the linear reconstruction
problem has a closed form analytical solution, while the nonlinear
inversion problem does not have a direct analytical solution and needs
to be solved using a numerical procedure. Linear reconstruction methods
can only be applied to the WPOA, while the POA already requires a
nonlinear reconstruction scheme. Unfortunately, in most realistic
experiments the WPOA does not apply.\
This seems to be a bit of a problem to me, as ideally we would find a
direct analytical solution to the inversion problem that is applicable
in realistic scenarios. I think that could be used to design a kernel
similar to how riCOM works, but this source claims that such a scheme
does not exist. Anyhow, I will try to summarize the techniques explained
in the book in order to gain more insight in the reconstruction
schemes.\

## Neutral microscope approximation

First, lets assume that the microscope perfectly magnifies and projects
the object wave function $\psi(\Vec{r})$ onto the detector, without any
aberrations. This is called a *neutral microscope*. The measured
intensity is the modulus squared of the wavefunction:
$$I(\vec{r})=\psi(\vec{r})\psi^*(\vec{r})$$ By Fourier transforming the
intensity, the product of the wavefunction and its complex conjugate
becomes a convolution (see equation
[\[eq:fourier_convolution_theorem\]](#eq:fourier_convolution_theorem){reference-type="ref"
reference="eq:fourier_convolution_theorem"}, this is the inverse of
that). In most conventions, $\vec{g}$ and $\vec{p}$ is used for the
spatial frequency instead of $\vec{k}$. I will try to be consistent and
use this notation.
$$I(\vec{g})=\int d\vec{p}\thinspace\psi(\vec{g}+\vec{p})\psi^*(\vec{\vec{p}})$$
This is a continuous Fourier transform, but since we are usually working
with crystalline samples which lead to discrete crystal reflections in
reciprocal space, the Fourier transform can be discretized:
$$I(\vec{g})=\sum_{\vec{p}}\psi(\vec{g}+\vec{p})\psi^*(\vec{\vec{p}})$$
This equation can interpreted as a summation of pairwise interferences
of beams. Every combination of beams with wavevectors $\vec{g}+\vec{p}$
and $\vec{p}$ will contribute to the Fourier coefficient $I(\vec{g})$,
as the complex conjugate of one of the beams leads to a subtraction of
the wavevectors.\
We can split this equation up into so-called *linear-interference* and
*nonlinear-interference* terms. The linear interference terms correspond
to interferences that include the undiffracted beam with $\vec{p}=0$ or
$\vec{g}+\vec{p}=0$. The non-linear interference terms correspond to
interferences between two diffracted beams. We can write the transformed
intensity as:
$$I(\vec{g}\neq0)=\psi(\vec{g})\psi(0)^*+\psi(0)\psi^*(-\vec{g})+\sum_{\vec{p}\neq0, \vec{p}\neq-\vec{g}}\psi(\vec{g}+\vec{p})\psi^*(\vec{\vec{p}})$$
In the WPOA, the intensity of the diffracted beams is very small and the
nonlinear terms can be neglected[^2]. This is however not a good
approximation for the POA. In the case of the WPOA, the intensity
becomes: $$\label{eq:intensity_g_linear}
    I_L(\vec{g}\neq0)=\psi(\vec{g})\psi(0)^*+\psi(0)\psi^*(-\vec{g})$$
Apparently, if we make this approximation of dropping the cross-terms, a
direct analytical inversion scheme for through-focal series
reconstruction becomes possible. If the cross-terms are kept, an
analytical solution would become impossible.\
If we assume the WPOA and assume that the sample is illuminated by a
wavefunction $\psi=1$, then we can Fourier transform the expression of
the object wavefunction (equation
[\[eq:weak_phase_object\]](#eq:weak_phase_object){reference-type="ref"
reference="eq:weak_phase_object"}) and substitute it into equation
[\[eq:intensity_g_linear\]](#eq:intensity_g_linear){reference-type="ref"
reference="eq:intensity_g_linear"}:
$$\psi(\Vec{r})=1+i\varphi(\Vec{r})\Leftrightarrow\psi(\Vec{g})=\delta(\vec{g})+i\varphi_p(\Vec{g})$$
$$\begin{aligned}
        I_L(\vec{g}\neq0)&=i\varphi_p(\vec{g})\left[1-i\varphi_p^*(0)\right]-i\varphi_p^*(-\vec{g})\left[1+i\varphi_p(0)\right]\\
        &\approx i\left[\varphi_p(\vec{g})-\varphi_p^*(-\vec{g})\right]
    \end{aligned}$$ The phase is real, so for the Fourier transform
$\varphi_p(\vec{g})=\varphi_p^*(-\vec{g})$ applies, and:
$$I_L(\vec{g}\neq0)=i\left[\varphi_p(\vec{g})+\varphi_p^*(-\vec{g})\right]=0$$
A perfect, aberrationless microscope will not show any contrast, as the
intensity of the image is not modulated. The phase difference cannot be
measured.

## Real microscope

Of course, a perfect microscope does not exist and aberrations will
cause the object wavefunction to not be perfectly projected onto the
detector. I have shown earlier that aberrations and defocus cause a
phase error $\chi(\vec{g})$. This is why the the Fourier transformed
wavefunction needs to be multiplied with the coherent phase transfer
function[^3] $t(\vec{g})=\exp{-i\chi(\vec{g})}$. The imaged wavefunction
$\psi_I$ in reciprocal space then becomes:
$$\psi_I(\vec{g})=\psi_O(\vec{g})\exp{-i\chi(\vec{g})}=\left[1+i\varphi_p(\vec{g})\right]\exp{-i\chi(\vec{g})}$$
Where $\psi_O$ represents the object wavefunction. In this case, if we
try again to work out equation
[\[eq:intensity_g_linear\]](#eq:intensity_g_linear){reference-type="ref"
reference="eq:intensity_g_linear"} we get: $$\begin{aligned}
            I_L(\vec{g}\neq0)&=\psi_I(\vec{g})\psi_I(0)^*+\psi_I(0)\psi_I^*(-\vec{g})\\
            &=i\varphi_p(\vec{g})\left[1-i\varphi_p^*(0)\right]\exp{-i\chi(\vec{g})}-i\varphi_p^*(-\vec{g})\left[1+i\varphi_p(0)\right]\exp{i\chi(-\vec{g})}\\
            &\approx i\varphi_p(\vec{g})\left[\exp{-i\chi(\vec{g})}-\exp{i\chi(-\vec{g})}\right]
    \end{aligned}$$ Since $\chi$ only depends on even powers of $g$ (see
equation
[\[eq:phase_error_simplified\]](#eq:phase_error_simplified){reference-type="ref"
reference="eq:phase_error_simplified"}), we get:
$$I(\vec{g}\neq0)= 2\varphi_p(\vec{g})\frac{\exp{i\chi(g)}-\exp{-i\chi(g)}}{2i}=2\varphi_p(\vec{g})\sin{\chi(g)}=2\FT{\Im{\psi(\vec{r})}}\sin{\chi(g)}$$
This is the same equation that has been derived earlier, in
[\[eq:simplified_result_focal_series_reconstruction\]](#eq:simplified_result_focal_series_reconstruction){reference-type="ref"
reference="eq:simplified_result_focal_series_reconstruction"}.\
Let us now generalize this for a more general wavefunction, where the
real part is also variable. We can do this by introducing a function
$\mu(\vec{r})$ that describes the variation of the real part of the
wavefunction. We then get: $$\begin{aligned}
        \psi_O(\vec{r})&=1+i\varphi(\vec{r})+\mu(\vec{r})\\
        \psi_O(\vec{g})&=\delta(\vec{g})+i\varphi_p(\vec{g})+\mu_p(\vec{g})\\
    \end{aligned}$$ The image wavefunction in reciprocal space is found
by multiplication with the coherent phase transfer function:
$$\psi_O(\vec{g})=\left[\delta(\vec{g})+i\varphi_p(\vec{g})+\mu_p(\vec{g})\right]\exp{-i\chi(g)}$$
Solving for the linear intensity by inserting this into equation
[\[eq:intensity_g_linear\]](#eq:intensity_g_linear){reference-type="ref"
reference="eq:intensity_g_linear"}, we get:
$$\label{eq:lin_image_formation_re_im}
    \begin{aligned}
            I_L(\vec{g}\neq0)=&\psi_I(\vec{g})\psi_I(0)^*+\psi_I(0)\psi_I^*(-\vec{g})\\
            =&\left[i\varphi_p(\vec{g})+\mu_p(\vec{g})\right]\left[1-i\varphi_p^*(0)+\mu_p^*(0)\right]\exp{-i\chi(g)}\\
            &+\left[-i\varphi_p^*(-\vec{g})+\mu_p^*(-\vec{g})\right]\left[1+i\varphi_p(0)+\mu_p(0)\right]\exp{i\chi(g)}\\
           \approx&\left[i\varphi_p(\vec{g})+\mu_p(\vec{g})\right]\exp{-i\chi(g)}+\left[-i\varphi_p(\vec{g})+\mu_p(\vec{g})\right]\exp{i\chi(g)}\\
           =&2\varphi_p(\vec{g})\frac{\exp{i\chi(g)}-\exp{-i\chi(g)}}{2i}+2\mu_p(\vec{g})\frac{\exp{i\chi(g)}+\exp{-i\chi(g)}}{2}\\
           =&2\varphi_p(\vec{g})\sin{\chi(g)}+2\mu_p(\vec{g})\cos{\chi(g)}\\
           =&2\FT{\Im{\psi_O(\vec{r})}}\sin{\chi(g)}+2\FT{\Re{\psi_O(\vec{r})}}\cos{\chi(g)}
    \end{aligned}$$ This expression can be interpreted as a projection.
At each position, the object wave has a real and an imaginary component,
which can be viewed as the two coordinates of a vector in the complex
plane. If we would ignore the Fourier transform operations, then this
equation describes the projection of the wavefunction-vector on an axis
in the complex plane, whose angle is given by the phase $\chi$. This
phase however, is dependent on the spatial frequency $\vec{g}$. That is
why we first need to Fourier transform, so that the wavefunction at each
spatial frequency is projected on the axis with the correct phase given
by $\chi(\vec{g}).$\
The way the linear intensity is formed is thus Fourier transform
$\rightarrow$ frequency-by-frequency projection on the complex axis
given by $\chi(\vec{g})$ $\rightarrow$ Inverse Fourier transform.\
The principle of focal series reconstruction is to vary the defocus,
which changes the projection angles given by $\chi(\vec{g})$, so that we
can "tomographically" solve for the wavefunction by combining the
projections at different angles. A focal series can be described as a
frequency-dispersive tomograph of the object wavefunction in the complex
number plane.\
Figure [3](#fig:tomographic_spirals){reference-type="ref"
reference="fig:tomographic_spirals"} shows the frequency-dependent
projection axes for different defocus values. For this visualization,
$C_s$ is set to $0$ so that $\chi(\vec{g})$ only contains the
defocus-dependent term.

![Tomographic view of the focus-variation principle in the 3D (Re, Im,
g) space, for defocus values of $0$ nm (a), $8$ nm (b), $16$ nm (c),
$32$ nm (d).
[@CarterWilliams2016TEM]](tomographic_spirals.png){#fig:tomographic_spirals
width="50%"}

## Reconstruction scheme

Williams and Carter approaches the linear inversion scheme in two
different ways. First, it is possible to prove that the linear
reconstruction problem has an analytical solution. Second, it gives an
explicit closed-form solution consisting of complex-valued filter
functions $f_{\Delta f_n}(\vec{g})$.

### General solution

For notational purposes, we will denote the Fourier transforms of the
real and imaginary parts of the (double of) object wave function as
follows: $$\begin{aligned}
        X_O(\vec{g})&:=2\FT{\Re{\psi(\vec{g})}}\\
        Y_O(\vec{g})&:=2\FT{\Im{\psi(\vec{g})}}\\
    \end{aligned}$$ Using these functions, we can write:
$$\label{eq:XOYO_linreconstruction}
    I_L(\vec{g}\neq0)=X_O(\vec{g})\cos{\chi(g)}+Y_O(\vec{g})\sin{\chi(g)}$$
The object wavefunction can be reconstructed from $X_O$ and $Y_O$:
$$\psi_O(\vec{r})=\frac{1}{2}\IFT{X_O(\vec{g})+iY_O(\vec{g})}$$ So this
means that if we would be able to calculate $X_O$ and $Y_O$ from a
defocus series, it would be possible to reconstruct the object wave
function.\
Lets assume that we take a defocus series of $N$ images, where the
$n$-th intensity $I_{L,n}$ corresponds to an image taken with defocus
$\Delta f_n$. The coherent phase transfer function for that image is
found from equation
[\[eq:phase_error_simplified\]](#eq:phase_error_simplified){reference-type="ref"
reference="eq:phase_error_simplified"}.
$$\chi_n(g):=2\pi\left(\frac{1}{4}C_{s3}\lambda^3g^4+\frac{1}{2}\lambda g^2\Delta f_n\right)$$
The frequency-dependent intensity for image $n$ can be described as:
$$\label{eq:XOYO_linreconstruction_defocusseries}
    I_{L,n}(\vec{g}\neq0)=X_O(\vec{g})\cos{\chi_n(g)}+Y_O(\vec{g})\sin{\chi_n(g)}$$
We can multiply this equation once with $\cos{\chi_n(g)}$ and once with
$\sin{\chi_n(g)}$: $$\begin{aligned}
        I_{L,n}(\vec{g})\cos{\chi_n(g)}&=X_O(\vec{g})\cos^2{\chi_n(g)}+Y_O(\vec{g})\sin{\chi_n(g)}\cos{\chi_n(g)}\\I_{L,n}(\vec{g})\sin{\chi_n(g)}&=X_O(\vec{g})\sin{\chi_n(g)}\cos{\chi_n(g)}+Y_O(\vec{g})\sin^2{\chi_n(g)}\\
    \end{aligned}$$ This starts to look like a system of linear
equations, but right now the coefficients are linearly dependent. This
is the mathematical way of saying that we need more than one image in
the defocus series in order to get a unique solution. If we sum both
equations over all $N$ images in the defocus series, the coefficients
become linearly independent and we get: $$\begin{aligned}
        \sum_n I_{L,n}(\vec{g})\cos{\chi_n(g)}&=X_O(\vec{g})\sum_n\cos^2{\chi_n(g)}+Y_O(\vec{g})\sum_n\sin{\chi_n(g)}\cos{\chi_n(g)}\\
        \sum_nI_{L,n}(\vec{g})\sin{\chi_n(g)}&=X_O(\vec{g})\sum_n\sin{\chi_n(g)}\cos{\chi_n(g)}+Y_O(\vec{g})\sum_n\sin^2{\chi_n(g)}\\
    \end{aligned}$$ Or in matrix form: $$\begin{pmatrix}
        \sum_n I_{L,n}(\vec{g})\cos{\chi_n(g)}\\
        \sum_n I_{L,n}(\vec{g})\sin{\chi_n(g)}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \sum_n \cos^2{\chi_n(g)} & \sum_n \sin{\chi_n(g)}\cos{\chi_n(g)}\\
        \sum_n \sin{\chi_n(g)}\cos{\chi_n(g)} & \sum_n \sin^2{\chi_n(g)}
    \end{pmatrix}
    \begin{pmatrix}
        X_O(\vec{g})\\
        Y_O(\vec{g})
    \end{pmatrix}$$ One of the challenges in focal series reconstruction
is choosing appropriate values for the defocus $\Delta f_n$ so that the
coefficients of this system do not lead to an ill-posed problem.

### Explicit solution for filter functions

We can write down an explicit analytical solution in the linear
approximation in the dorm of a set of complex-valued filter functions
$f_{\Delta f_n}(\vec{g})$. These functions can be used to reconstruct
the wavefunction from a defocus series $I_{L,n}(\vec{g})$ as follows:
$$\psi_O(\vec{g}\neq0)=\sum_{n=1}^Nf_{\Delta f_n}(\vec{g})I_{L,n}(\vec{g})$$
Fourier transforming this back to real space, we find that the filter
functions correspond to convolutional filters or *kernels*
$$\psi_O(\vec{r})=C+\sum_{n=1}^Nf_{\Delta f_n}(\vec{r})\ast I_{L,n}(\vec{r})$$
I added the constant $C$ because the value for $\psi(\vec{g}=0)$ is not
known.\
We can begin with the expression for linear image formation for a
defocus value $f_{\Delta f_n}$, as shown in equation
[\[eq:intensity_g_linear\]](#eq:intensity_g_linear){reference-type="ref"
reference="eq:intensity_g_linear"}, and substituting the image
wavefunction for the object wavefunction multiplied with the coherent
phase transfer function.
$$I_{L,n}(\vec{g}\neq0)=\psi_O(\vec{g})\psi_O(0)^*\exp{-i\chi_n(g)}+\psi_O(0)\psi_O^*(-\vec{g})\exp{i\chi_n(g)}$$
We can choose a global phase for the wavefunction, since this does not
change the observables. We can choose this phase so that $\psi_O(0)$ is
real, and $\psi_O(0)=\psi_O^*(0)$. This is then only a scale factor for
the wavefunction, and can be dropped from the equation:
$$I_{L,n}(\vec{g}\neq0)=\psi_O(\vec{g})\exp{-i\chi_n(g)}+\psi_O^*(-\vec{g})\exp{i\chi_n(g)}$$
We can now multiply this equation once with $\exp{-i\chi_n(g)}$ and once
with $\exp{i\chi_n(g)}$: $$\begin{aligned}
        I_{L,n}(\vec{g}\neq0)\exp{-i\chi_n(g)}&=\psi_O(\vec{g})\exp{-2i\chi_n(g)}+\psi_O^*(-\vec{g})\\
        I_{L,n}(\vec{g}\neq0)\exp{i\chi_n(g)}&=\psi_O(\vec{g})+\psi_O^*(-\vec{g})\exp{2i\chi_n(g)}\\
    \end{aligned}$$ Summing over the equation for every defocus value
gives: $$\begin{aligned}
        \sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\exp{-i\chi_n(g)}&=\psi_O(\vec{g})\sum_{n=1}^N\exp{-2i\chi_n(g)}+N\psi_O^*(-\vec{g})\\
        \sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\exp{i\chi_n(g)}&=N\psi_O(\vec{g})+\psi_O^*(-\vec{g})\sum_{n=1}^N\exp{2i\chi_n(g)}\\
    \end{aligned}$$ This system of equations for $\psi_O(\vec{g})$ and
$\psi_O^*(-\vec{g})$ can be solved analytically. Since we only need
$\psi_O(\vec{g})$, we can multiply the first equation with
$1/N\sum_{n=1}^N\exp{2i\chi_n(g)}$ and subtract the equations to
eliminate $\psi_O^*(-\vec{g})$: $$\begin{aligned}
        \frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\exp{-i\chi_n(g)}-\sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\exp{i\chi_n(g)}&\\
        =\frac{1}{N}\psi_O(\vec{g})\sum_{n=1}^N\exp{2i\chi_n(g)}\sum_{n=1}^N\exp{-2i\chi_n(g)}-N\psi_O(\vec{g})&\\
        \Leftrightarrow\sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\left[\exp{-i\chi_n(g)}\left[\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\right]-\exp{i\chi_n(g)}\right]&\\
        =\psi_O(\vec{g})\left[\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\sum_{n=1}^N\exp{-2i\chi_n(g)}-N\right]&\\
         \Leftrightarrow\psi_O(\vec{g})=\sum_{n=1}^NI_{L,n}(\vec{g}\neq0)\frac{\exp{i\chi_n(g)}-\exp{-i\chi_n(g)}\left[\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\right]}{N-\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\sum_{n=1}^N\exp{-2i\chi_n(g)}}
    \end{aligned}$$ So the filter functions are:
$$f_{\Delta f_n}(g)=\frac{\exp{i\chi_n(g)}-\exp{-i\chi_n(g)}\left[\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\right]}{N-\frac{1}{N}\sum_{n=1}^N\exp{2i\chi_n(g)}\sum_{n=1}^N\exp{-2i\chi_n(g)}}$$
For large N the second terms in the numerator and denominator can be
neglected and this reduces to:
$$f_{\Delta f_n}(g)=\frac{1}{N}\exp{i\chi_n(g)}$$ These functions are
exactly what we want, and can be applied for event-driven
reconstruction. Every detected electron at a position $\vec{r}'$ and a
defocus value $\Delta f_n$ can be considered a delta function
$I(\vec{r})=\delta(\vec{r}-\vec{r}')$, and will have a contribution
$f_{\Delta f_n}(\vec{r}-\vec{r}')$ to the wavefunction reconstruction,
where $f_{\Delta f_n}(\vec{r})=\IFT{f_{\Delta f_n}(\vec{g})}$.\
The challenge now is trying to find a similar approach that is
applicable in realistic scenarios, where the WPOA does not hold and
higher powers of $\varphi$ cannot be neglected. Williams and Carter
however states that no direct analytical solution to the nonlinear
inversion problem exist.\
However, I don't think that necessarily means that an event-driven
acquisition technique is impossible, just that a closed-form expression
for a kernel that takes nonlinear terms into account is not possible.

[^1]: In reality, other aberrations such as astigmatism and coma are
    also present and break the rotational symmetry, causing $\alpha_x$
    and $\alpha_y$ to contribute independently to the aberration.

[^2]: I would like to better understand how this follows from the
    expansion of the exponential term in equation
    [\[eq:weak_phase_object\]](#eq:weak_phase_object){reference-type="ref"
    reference="eq:weak_phase_object"}. You can probably prove that the
    nonlinear terms are quadratic in $V$?

[^3]: In Williams and Carter, $t(\vec{g})$ is used for the coherent
    phase transfer function instead of $H_0(\vec{g})$
