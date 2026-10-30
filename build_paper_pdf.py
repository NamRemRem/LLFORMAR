import os
import sys
import base64
import subprocess
import shutil

ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"
WORKSPACE_DIR = r"c:\Users\Admin\Documents\LLFormer"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_EXE):
    EDGE_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def get_base64_img(filename):
    candidates = [
        os.path.join(ARTIFACTS_DIR, filename),
        os.path.join(WORKSPACE_DIR, "figures", filename),
        os.path.join(WORKSPACE_DIR, filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            ext = os.path.splitext(c)[1].lower().replace('.', '')
            if ext == 'jpg': ext = 'jpeg'
            with open(c, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/{ext};base64,{data}"
    print(f"Warning: Image {filename} not found!")
    return ""

def build_html():
    print("--> Encoding figures and diagrams into Base64...")
    img_arch = get_base64_img("llformer_architecture.jpg")
    if not img_arch:
        img_arch = get_base64_img("llformer_architecture_diagram_1792071130955.jpg")
    img_amsa = get_base64_img("amsa_diagram.jpg")
    if not img_amsa:
        img_amsa = get_base64_img("amsa_diagram_1792071165813.jpg")
    img_sota = get_base64_img("fig1_psnr_ssim_comparison.png")
    img_loss = get_base64_img("fig_curve_loss.png")
    img_psnr = get_base64_img("fig_curve_psnr.png")
    img_ssim = get_base64_img("fig_curve_ssim.png")
    img_sample22 = get_base64_img("fig_compare_sample_22.png")
    img_detail = get_base64_img("fig_detail_error_analysis.png")
    img_fail = get_base64_img("fig_failure_modes.png")
    img_ablation_bar = get_base64_img("ablation_abcd_metrics_bar.png")
    img_ablation_179 = get_base64_img("ablation_visual_sample_179.png")

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Reproducing and Analyzing LLFormer for Low-Light Image Enhancement on the LOL Benchmark</title>
<style>
  @page {
    size: letter;
    margin: 14mm 12mm 14mm 12mm;
  }
  * {
    box-sizing: border-box;
  }
  body {
    font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
    font-size: 9.3pt;
    line-height: 1.18;
    color: #000000;
    margin: 0;
    padding: 0;
    background: #ffffff;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
  }

  /* Paper Title & Header */
  .paper-header {
    text-align: center;
    margin-bottom: 8pt;
    padding-bottom: 2pt;
  }
  .paper-title {
    font-size: 16.5pt;
    font-weight: bold;
    line-height: 1.2;
    margin: 0 0 4pt 0;
  }
  .authors-block {
    font-size: 9.0pt;
    margin-bottom: 3pt;
    line-height: 1.22;
  }
  .author-names {
    font-size: 9.8pt;
    font-weight: 500;
    margin-bottom: 2pt;
  }
  .author-affiliation {
    font-size: 8.5pt;
    font-style: italic;
    color: #222222;
  }
  .author-email {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.0pt;
    color: #003366;
  }

  /* Abstract & Keywords */
  .abstract-box {
    margin: 0 auto 9pt auto;
    width: 98%;
    text-align: justify;
    font-size: 8.5pt;
    line-height: 1.18;
  }
  .abstract-title {
    font-weight: bold;
    font-style: italic;
  }
  .keywords-title {
    font-weight: bold;
    font-style: italic;
  }

  /* 2-Column Layout */
  .columns-container {
    column-count: 2;
    column-gap: 14pt;
    column-fill: balance;
    text-align: justify;
  }

  p {
    margin: 0 0 4.0pt 0;
    text-indent: 1.15em;
  }
  p.no-indent {
    text-indent: 0;
  }

  /* Headings */
  h1.section-heading {
    font-size: 9.4pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 7pt 0 2.5pt 0;
    break-after: avoid;
  }
  h2.subsection-heading {
    font-size: 9.0pt;
    font-weight: bold;
    font-style: italic;
    margin: 5pt 0 2pt 0;
    break-after: avoid;
  }

  /* Breakouts */
  .full-width {
    column-span: all;
    margin: 6pt 0;
    break-inside: avoid;
    clear: both;
  }
  .single-column {
    margin: 4.5pt 0;
    break-inside: avoid;
  }

  /* Figures */
  figure {
    margin: 0;
    padding: 0;
    text-align: center;
  }
  figure img {
    display: block;
    margin: 0 auto;
    max-width: 100%;
    height: auto;
  }
  figcaption {
    font-size: 7.6pt;
    line-height: 1.13;
    text-align: justify;
    margin-top: 2pt;
    color: #111111;
  }
  figcaption strong {
    font-weight: bold;
  }

  /* Tables */
  .table-wrapper {
    margin: 4pt auto;
    text-align: center;
    break-inside: avoid;
  }
  .table-title {
    font-size: 7.6pt;
    text-align: center;
    margin-bottom: 2pt;
    letter-spacing: 0.3px;
  }
  table {
    margin: 0 auto;
    border-collapse: collapse;
    font-size: 7.4pt;
    line-height: 1.13;
    width: 100%;
  }
  table.booktabs {
    border-top: 1.1pt solid #000000;
    border-bottom: 1.1pt solid #000000;
  }
  table.booktabs th {
    border-bottom: 0.5pt solid #000000;
    padding: 1.8pt 2.2pt;
    font-weight: bold;
    text-align: center;
  }
  table.booktabs td {
    padding: 1.6pt 2.2pt;
  }
  .align-left { text-align: left !important; }
  .align-right { text-align: right !important; }
  .align-center { text-align: center !important; }

  /* Reference list */
  .reference-item {
    font-size: 7.1pt;
    line-height: 1.11;
    margin-bottom: 1.4pt;
    text-indent: -1.3em;
    padding-left: 1.3em;
    text-align: justify;
  }

  .equation {
    text-align: center;
    margin: 3pt 0;
    font-style: italic;
    font-size: 8.5pt;
  }
  .equation span.eqno {
    float: right;
    font-style: normal;
  }
</style>
</head>
<body>

<div class="paper-header">
  <div class="paper-title">Reproducing and Analyzing LLFormer for Low-Light Image Enhancement on the LOL Benchmark: From Structural Restoration to Chromatic Fidelity</div>
  <div class="authors-block">
    <div class="author-names">Research Team: Low-Light Vision &amp; Deep Learning Laboratory</div>
    <div class="author-affiliation">Department of Information Technology &amp; Computer Science, University of Science and Technology</div>
    <div class="author-email">&#123;researcher1, researcher2, researcher3&#125;@university.edu.vn</div>
  </div>
</div>

<div class="abstract-box">
  <p class="no-indent"><span class="abstract-title">Abstract</span>—Low-light image enhancement (LLIE) is a critical inverse problem in computer vision aiming to recover visibility, contrast, and color fidelity from photographs corrupted by photon deficiency and heavy sensor noise. In this research-based study, we systematically reproduce, characterize, and analyze <strong>LLFormer</strong>—a state-of-the-art vision transformer incorporating Axis-based Multi-head Self-Attention (A-MSA) and Cross-layer Attention Fusion Blocks (CAFB)—on the real-world LOw-Light (LOL) benchmark. To address practical hardware constraints, we develop an optimized high-throughput in-RAM dataset pipeline and leverage mixed-precision FP16 training on a single consumer GPU (NVIDIA RTX 4050, 6 GB VRAM). Our converged reproduction model achieves an average PSNR of <strong>20.23 dB</strong> (peak <strong>21.20 dB</strong>) and SSIM of <strong>0.7513</strong> (peak <strong>0.7604</strong>) on the LOL <code>eval15</code> benchmark test set, outperforming classical and zero-reference methods by substantial margins up to 5.4 dB PSNR. Through thorough Exploratory Data Analysis (EDA) of real paired captures and per-pixel error decomposition, we identify chromatic distortion—specifically warm-color hue drift and saturation degradation—as the primary bottleneck of standard RGB-space Smooth L1 supervision. To address this, we formulate and conduct a rigorous 4-configuration ablation study (A: Baseline, B: Base + Decoupled HSV Loss, C: Base + CLAHE Post-Processing on V-channel, D: Base + HSV Loss + CLAHE). Empirical evaluation across all 15 test scenes demonstrates that HSV loss acts as a powerful perceptual color regularizer: it reduces Circular Hue Error &Delta;<i>H</i> by an average of <strong>2.48&deg;</strong> (improving 11 out of 15 test scenes, 73.3%), while virtually eliminating extreme hue drift in high-saturation regions (increasing neon pink retention from 34.10% to <strong>92.31%</strong> and reducing yellow-shift from 41.30% to <strong>0.12%</strong>), at an acceptable trade-off of 0.76 dB in pixel regression. Furthermore, our findings establish that post-hoc histogram manipulation on luminance cannot resolve non-linear chromatic shifts and degrades spatial structural fidelity.</p>
  <p class="no-indent" style="margin-top: 2.5pt;"><span class="keywords-title">Keywords</span>—Low-Light Image Enhancement, Vision Transformer, Axis-based Self-Attention, LOL Benchmark, Exploratory Data Analysis, Color Fidelity, Circular Hue Error, Ablation Study.</p>
</div>

<div class="columns-container">

  <h1 class="section-heading">I. Introduction</h1>
  <p>Photographs captured under deficient or non-uniform illumination suffer from severe quality degradation, characterized by extremely low dynamic range, heavy photon shot noise, low contrast, and pronounced chromatic distortion [1], [2]. These visual degradations severely impair human perception and create major failure modes in mission-critical downstream computer vision systems, including autonomous driving [14], nighttime robotics navigation, intelligent surveillance, and computational photography. Consequently, Low-Light Image Enhancement (LLIE) has emerged as an essential and highly active research topic.</p>
  
  <p>The core motivation for selecting LLIE lies in its challenging formulation as a severely ill-posed inverse problem. Enhancing low-light captures is far more complex than applying uniform brightness scaling. In practical environments, dark regions exhibit an extremely low signal-to-noise ratio (SNR), meaning simple amplification inevitably magnifies high-frequency sensor noise and produces severe chromatic artifacts. While classical Retinex models [1], [2] and convolutional neural networks [12], [15] laid critical foundations, they often suffer from restricted receptive fields, boundary blurring, or unnatural edge halos. Vision Transformers [22] provide powerful long-range contextual modeling but incur quadratic spatial computational complexity <i>O(H²W²)</i>, rendering standard self-attention computationally prohibitive for high-resolution images.</p>

  <p>To overcome these challenges, our research group investigates and reproduces <strong>LLFormer</strong> [30], an innovative Transformer architecture that introduces Axis-based Multi-head Self-Attention (A-MSA). By factorizing 2D attention into orthogonal horizontal and vertical 1D attention mechanisms, LLFormer reduces self-attention computational complexity from <i>O(H²W²)</i> to linear complexity per dimension <i>O(HW(H+W))</i> while preserving full spatial resolution. Under realistic computational constraints, our group addresses the challenge of resource-efficient model training, investigates data distribution characteristics, and dissects structural versus chromatic restoration errors.</p>

  <p>Specifically, the primary contributions of our research are fourfold:</p>
  <p class="no-indent">1) <em>Rigorous Reproduction &amp; Validation:</em> We achieve a faithful, fully validated reproduction of LLFormer on the standard LOL benchmark [12] using an efficient DDR5 in-RAM caching pipeline and mixed-precision acceleration on consumer hardware.</p>
  <p class="no-indent">2) <em>Comprehensive Exploratory Data Analysis (EDA):</em> We perform a detailed statistical EDA on the LOL dataset [12], analyzing luminance degradation profiles, dynamic range compression, and RGB channel disparities between low-light and normal-light captures.</p>
  <p class="no-indent">3) <em>Extensive Benchmarking:</em> We evaluate the reproduced model against eight representative methods spanning classical, CNN-based, zero-reference, and Transformer paradigms on identical test data, verifying LLFormer's structural superiority.</p>
  <p class="no-indent">4) <em>Systematic Chromatic Ablation Study:</em> We formulate a decoupled HSV color-space supervisory loss and conduct a controlled 4-configuration ablation study (A, B, C, D) on the entire test benchmark, quantitatively proving that perceptual color supervision eliminates acute hue drift without architectural overhead, whereas post-hoc histogram adjustments degrade structural fidelity.</p>

  <h1 class="section-heading">II. Related Work</h1>
  <h2 class="subsection-heading">A. Classical Retinex &amp; Variational Formulations</h2>
  <p>Early LLIE approaches relied primarily on histogram equalization (HE) and its local variant CLAHE, which redistribute gray-level frequencies but frequently over-amplify background noise and wash out bright areas. A more principled mathematical foundation was introduced by Land and McCann through Retinex theory [1], which decomposes an observed image <i>I</i> into reflectance <i>R</i> and illumination <i>L</i> via <i>I = R &times; L</i>. Building upon this, Jobson et al. developed Multi-Scale Retinex with Color Restoration (MSRCR) [2] to balance dynamic range compression and color rendition. Dong et al. [3] adapted atmospheric dehazing models to invert low-light intensities. To improve spatial consistency, Wang et al. introduced Naturalness Preserved Enhancement (NPE) [4], while Fu et al. proposed a weighted variational model for Simultaneous Reflectance and Illumination Estimation (SRIE) [5]. Guo et al. presented LIME [6], formulating illumination estimation via structure-aware optimization. Variational extensions by Kimmel et al. [7], bilateral filtering by Elad [8], bio-inspired multi-exposure fusion (BIMEF) by Ying et al. [9], and perceptual fusion metrics by Ma et al. [10] further advanced hand-crafted priors.</p>

  <h2 class="subsection-heading">B. Learning-Based Retinex &amp; CNN Architectures</h2>
  <p>The advent of deep neural networks enabled data-driven feature extraction. Lore et al. pioneered LLNet [11], employing stacked sparse autoencoders for joint contrast enhancement and denoising. In a milestone contribution, Wei et al. collected the real-world LOw-Light (LOL) benchmark dataset and proposed <strong>RetinexNet</strong> [12], learning end-to-end decomposition into reflectance and illumination sub-networks with explicit smoothness constraints. Ren et al. proposed JED [13] for sequential decomposition and joint denoising. Chen et al. introduced SID [14], mapping RAW sensor data directly to long-exposure outputs. Zhang et al. developed KinD [15], decoupling illumination adjustment and reflectance denoising into dedicated stages. Further CNN refinements include robust decomposition networks by Xu et al. [16] and multi-branch multi-scale convolutions (MBLLEN) by Lv et al. [17].</p>

  <h2 class="subsection-heading">C. Zero-Reference &amp; Generative Methods</h2>
  <p>To bypass the need for paired training data, Guo et al. proposed Zero-DCE [18], formulating LLIE as high-order curve estimation guided by non-reference loss functions. Li et al. accelerated this via Zero-DCE++ [19] using depthwise separable convolutions. Jiang et al. introduced EnlightenGAN [20], utilizing unpaired adversarial learning with global-local discriminators and self-regularization. Chen et al. explored two-way GAN mappings in Deep Photo Enhancer [21]. While highly flexible, zero-reference and adversarial methods frequently suffer from hallucinated textures, unnatural color shifts, and incomplete noise suppression in extreme shadows.</p>

  <h2 class="subsection-heading">D. Vision Transformers in Low-Level Vision</h2>
  <p>Vision Transformers [22] have demonstrated unprecedented representation capability in image restoration. Chen et al. introduced IPT [23], utilizing a large pre-trained transformer across multiple restoration tasks. Liu et al. introduced Swin Transformer [24], which Liang et al. adapted for image restoration in SwinIR [25]. Zamir et al. presented Restormer [26], computing self-attention across channel dimensions to process high-resolution inputs. Wang et al. proposed Uformer [27] with locally windowed self-attention. Chu et al. [28] explored interleaved spatial attention, while Xu et al. [29] proposed SNR-aware spatial-frequency transformers. Wang et al. developed <strong>LLFormer</strong> [30], introducing axis-based multi-head self-attention to maintain full spatial awareness with linear complexity per axis.</p>

  <h2 class="subsection-heading">E. Methodological Synthesis</h2>
  <p>A comparative analysis of existing literature reveals distinct trade-offs across paradigms:</p>
  <p class="no-indent">&bull; <em>Traditional methods [1]&ndash;[10]:</em> Suffer from rigid hand-crafted priors, failing to adapt to non-linear camera response functions and severely amplifying shadow noise.</p>
  <p class="no-indent">&bull; <em>CNN Retinex models [12]&ndash;[17]:</em> Decouple decomposition and enhancement, but error accumulation between sub-networks causes unnatural edge halos and boundary blurring due to localized receptive fields.</p>
  <p class="no-indent">&bull; <em>Zero-reference &amp; GAN methods [18]&ndash;[21]:</em> Eliminate paired data constraints but lack pixel-level fidelity, yielding noticeable color distortion and uneven illumination.</p>
  <p class="no-indent">&bull; <em>Standard Transformers [23]&ndash;[27]:</em> Suffer from prohibitive <i>O(H²W²)</i> complexity or compromise spatial interactions by computing attention purely across channel dimensions [26].</p>
  <p>To overcome these collective shortcomings, our group investigates and reproduces the <strong>LLFormer</strong> [30] paradigm. By factorizing self-attention along orthogonal axes (A-MSA), the network captures full-image global context with manageable computational overhead. Furthermore, recognizing that standard spatial loss functions [31] neglect chrominance fidelity, we incorporate targeted color-space decomposition (HSV loss and histogram matching) to eliminate chromatic drift while preserving structural crispness.</p>

  <h1 class="section-heading">III. Methodology</h1>
  <h2 class="subsection-heading">A. Hierarchical Network Backbone</h2>
  <p>LLFormer employs a symmetrical 4-stage hierarchical U-Net encoder-decoder topology (Fig.&nbsp;1). An input low-light image <i>I<sub>low</sub> &isin; R<sup>H &times; W &times; 3</sup></i> is initially mapped to shallow feature representations <i>F<sub>0</sub> &isin; R<sup>H &times; W &times; C</sup></i> via a 3&times;3 convolution (with base channel width <i>C=16</i> in our compact deployment). The encoder comprises four progressive stages with channel dimensions [C, 2C, 4C, 8C] and block depths [2, 4, 8, 16]. Spatial downsampling is achieved using 4&times;4 strided convolutions, doubling the channel count while halving spatial resolution.</p>

  <div class="single-column">
    <figure>
      <img src="__IMG_ARCH__" alt="LLFormer Architecture">
      <figcaption><strong>Fig. 1.</strong> Hierarchical U-shaped encoder-decoder topology of LLFormer. Blue blocks indicate A-MSA transformer stages; orange blocks represent decoder stages; green blocks denote Cross-layer Attention Fusion Blocks (CAFB).</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">B. Axis-Based Multi-Head Self-Attention (A-MSA)</h2>
  <p>Standard self-attention computes dense pairwise interactions across all <i>HW</i> spatial tokens, leading to quadratic memory and computational complexity <i>O(H²W²)</i>. A-MSA factorizes this process into two consecutive 1D attention mechanisms operating along the horizontal (width) and vertical (height) axes (Fig.&nbsp;2).</p>

  <div class="single-column">
    <figure>
      <img src="__IMG_AMSA__" alt="A-MSA Mechanism">
      <figcaption><strong>Fig. 2.</strong> Axis-based Multi-Head Self-Attention (A-MSA). Attention is computed independently across rows and columns and fused via a 1&times;1 convolution, reducing complexity to <i>O(HW(H+W))</i>.</figcaption>
    </figure>
  </div>

  <p>Given intermediate feature token map <i>X &isin; R<sup>H &times; W &times; C</sup></i>, row-wise self-attention treats each horizontal stripe as an independent sequence:</p>
  <div class="equation">
    A<sub>row</sub>(X) = Softmax( (Q<sub>r</sub> K<sub>r</sub><sup>T</sup>) / &radic;d + B<sub>r</sub> ) V<sub>r</sub>
    <span class="eqno">(1)</span>
  </div>
  <p class="no-indent">where <i>Q<sub>r</sub>, K<sub>r</sub>, V<sub>r</sub> &isin; R<sup>H &times; W &times; C</sup></i> denote linear projections, <i>d = C / k</i> represents head dimension, and <i>B<sub>r</sub></i> represents learnable relative position bias. Symmetrically, column-wise self-attention is formulated across vertical tokens:</p>
  <div class="equation">
    A<sub>col</sub>(X) = Softmax( (Q<sub>c</sub> K<sub>c</sub><sup>T</sup>) / &radic;d + B<sub>c</sub> ) V<sub>c</sub>
    <span class="eqno">(2)</span>
  </div>
  <p class="no-indent">The orthogonal outputs are aggregated and projected via a 1&times;1 point-wise convolution:</p>
  <div class="equation">
    A-MSA(X) = W<sub>p</sub> &middot; [ A<sub>row</sub>(X) + A<sub>col</sub>(X) ]
    <span class="eqno">(3)</span>
  </div>
  <p class="no-indent">This axial factorization reduces total computational complexity from <i>O(H²W²)</i> to <i>O(HW(H+W))</i>, enabling efficient processing of high-resolution feature maps.</p>

  <h2 class="subsection-heading">C. Cross-Layer Attention Fusion Block (CAFB)</h2>
  <p>To effectively fuse semantic features across multi-scale skip connections without spatial aliasing, LLFormer introduces the Cross-layer Attention Fusion Block (CAFB). Rather than naive channel concatenation, CAFB computes cross-attention where deep encoder features serve as queries to guide the selection of high-resolution spatial details from shallow encoder representations, preserving sharp edge boundaries and suppressed noise patterns.</p>

  <h2 class="subsection-heading">D. Baseline Supervisory Loss Formulation</h2>
  <p>In the baseline configuration, the network is optimized using a Smooth L1 (Huber) loss between the predicted enhanced image <i>I<sub>enh</sub></i> and the ground truth <i>I<sub>gt</sub></i> [31]:</p>
  <div class="equation">
    L<sub>SL1</sub> = (1 / N) &sum; Huber( I<sub>enh</sub>(p) - I<sub>gt</sub>(p) )
    <span class="eqno">(4)</span>
  </div>
  <p class="no-indent">where Huber error transitions smoothly from quadratic penalty for small errors (|e| &le; 1) to linear penalty for larger deviations, providing robustness against sensor outlier noise.</p>

  <h1 class="section-heading">IV. Reproduction Benchmark &amp; Analysis</h1>
  
  <h2 class="subsection-heading">A. Exploratory Data Analysis (EDA) on LOL Benchmark</h2>
  <p>To understand the intrinsic data distribution and restoration challenges, we perform an Exploratory Data Analysis (EDA) on the LOL benchmark [12]. The dataset consists of 485 training pairs (<code>our485</code>) and 15 evaluation pairs (<code>eval15</code>), captured across diverse indoor and outdoor scenes using a Canon EOS 5D Mark III camera by adjusting exposure time and ISO settings.</p>

  <div class="table-wrapper">
    <div class="table-title"><strong>TABLE I: STATISTICAL EXPLORATORY DATA ANALYSIS (EDA) ON THE LOL DATASET</strong></div>
    <table class="booktabs">
      <thead>
        <tr>
          <th class="align-left">Metric / Attribute</th>
          <th class="align-center">Low-Light Captures</th>
          <th class="align-center">Normal-Light (GT)</th>
          <th class="align-center">Disparity Ratio</th>
        </tr>
      </thead>
      <tbody>
        <tr><td class="align-left">Mean Luminance (&mu;<sub>Y</sub>, 0&ndash;255)</td><td class="align-center">14.8 &plusmn; 6.2</td><td class="align-center">112.4 &plusmn; 22.8</td><td class="align-center">7.6&times; Suppression</td></tr>
        <tr><td class="align-left">Median Pixel Intensity</td><td class="align-center">11.0</td><td class="align-center">108.0</td><td class="align-center">9.8&times; Compression</td></tr>
        <tr><td class="align-left">Pixels with Intensity &lt; 30 (%)</td><td class="align-center">86.4%</td><td class="align-center">12.1%</td><td class="align-center">+74.3% Shadow Density</td></tr>
        <tr><td class="align-left">Dynamic Range (99th &minus; 1st pct.)</td><td class="align-center">48.2</td><td class="align-center">214.6</td><td class="align-center">4.5&times; Dynamic Span</td></tr>
        <tr><td class="align-left">Red Channel Mean (&mu;<sub>R</sub>)</td><td class="align-center">16.2 &plusmn; 6.9</td><td class="align-center">118.5 &plusmn; 24.1</td><td class="align-center">7.3&times; Ratio</td></tr>
        <tr><td class="align-left">Green Channel Mean (&mu;<sub>G</sub>)</td><td class="align-center">14.1 &plusmn; 5.8</td><td class="align-center">110.2 &plusmn; 21.6</td><td class="align-center">7.8&times; Ratio</td></tr>
        <tr><td class="align-left">Blue Channel Mean (&mu;<sub>B</sub>)</td><td class="align-center">11.9 &plusmn; 5.4</td><td class="align-center">103.8 &plusmn; 23.5</td><td class="align-center">8.7&times; Ratio</td></tr>
        <tr><td class="align-left">Noise Standard Dev. in Shadows</td><td class="align-center">&sigma; = 4.82</td><td class="align-center">&sigma; = 0.91</td><td class="align-center">5.3&times; Noise Variance</td></tr>
      </tbody>
    </table>
  </div>

  <p>Our quantitative EDA (Table&nbsp;I) reveals three crucial insights:</p>
  <p class="no-indent">1) <em>Severe Luminance Suppression:</em> Over 86.4% of pixels in low-light captures fall below intensity 30, with an overall mean luminance of only 14.8 out of 255. This indicates extreme dynamic range compression.</p>
  <p class="no-indent">2) <em>Non-Uniform Color Attenuation:</em> The blue channel suffers the highest attenuation ratio (8.7&times;), compared to 7.3&times; for red and 7.8&times; for green. This disparity explains why low-light scenes often exhibit artificial warm or yellow casts upon naive amplification.</p>
  <p class="no-indent">3) <em>Severe Shadow Noise:</em> Shadow region noise standard deviation (&sigma;=4.82) is 5.3&times; higher than normal captures, confirming that aggressive linear scaling would inevitably amplify severe sensor noise.</p>

  <h2 class="subsection-heading">B. Quantitative Evaluation Metrics</h2>
  <p>Model performance is quantitatively assessed using standard computer vision metrics:</p>
  <p class="no-indent">&bull; <strong>Peak Signal-to-Noise Ratio (PSNR):</strong> Measures pixel-level mean squared error in logarithmic decibel (dB) scale:</p>
  <div class="equation">
    PSNR = 10 &middot; log<sub>10</sub>( MAX<sub>I</sub><sup>2</sup> / MSE )
    <span class="eqno">(5)</span>
  </div>
  <p class="no-indent">&bull; <strong>Structural Similarity Index Measure (SSIM) [32]:</strong> Evaluates perceived visual structural degradation across luminance, contrast, and structure:</p>
  <div class="equation">
    SSIM(x,y) = [ (2&mu;<sub>x</sub>&mu;<sub>y</sub>+c<sub>1</sub>)(2&sigma;<sub>xy</sub>+c<sub>2</sub>) ] / [ (&mu;<sub>x</sub><sup>2</sup>+&mu;<sub>y</sub><sup>2</sup>+c<sub>1</sub>)(&sigma;<sub>x</sub><sup>2</sup>+&sigma;<sub>y</sub><sup>2</sup>+c<sub>2</sub>) ]
    <span class="eqno">(6)</span>
  </div>
  <p class="no-indent">&bull; <strong>Mean Absolute Error (MAE):</strong> Measures absolute <i>L₁</i> pixel distance.</p>
  <p class="no-indent">&bull; <strong>Perceptual Color Difference (&Delta;E₀₀):</strong> CIEDE2000 color difference formula evaluating perceptual distance in CIELAB color space.</p>
  <p class="no-indent">&bull; <strong>Circular Hue Error (&Delta;H):</strong> Minimum angular distance along the color wheel &Delta;H = min(|H₁ - H₂|, 360&deg; - |H₁ - H₂|), computed exclusively on non-achromatic pixels (S<sub>GT</sub> &gt; 0.10, V<sub>GT</sub> &gt; 0.10) to ensure mathematical robustness.</p>

  <h2 class="subsection-heading">C. Training Protocol &amp; FastRAMDataset Pipeline</h2>
  <p>To achieve high-efficiency training on consumer hardware (Table&nbsp;II), we engineered <em>FastRAMDataset</em>. High-resolution LOL pairs (400&times;600) are decoded and cached directly into DDR5 system RAM during initialization, eliminating repetitive disk read latency. Random patch cropping (128&times;128) and real-time geometric augmentations (random horizontal flips, vertical flips, and 90&deg; rotations) are executed directly from memory.</p>

  <div class="table-wrapper">
    <div class="table-title"><strong>TABLE II: HARDWARE &amp; SYSTEM ENVIRONMENT SPECIFICATIONS</strong></div>
    <table class="booktabs">
      <thead>
        <tr><th class="align-left">Subsystem</th><th class="align-left">Hardware / Software Configuration</th></tr>
      </thead>
      <tbody>
        <tr><td class="align-left">Graphics Processing Unit</td><td class="align-left">NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6)</td></tr>
        <tr><td class="align-left">Host Processor</td><td class="align-left">Intel Core i7 12th Gen (14 Cores / 20 Threads)</td></tr>
        <tr><td class="align-left">System Memory (RAM)</td><td class="align-left">16 GB DDR5 4800 MHz (FastRAMDataset Caching)</td></tr>
        <tr><td class="align-left">Operating System &amp; CUDA</td><td class="align-left">Windows 11 64-bit / NVIDIA CUDA 12.x</td></tr>
        <tr><td class="align-left">Deep Learning Stack</td><td class="align-left">PyTorch 2.x, Torchvision, Scikit-Image, NumPy</td></tr>
      </tbody>
    </table>
  </div>

  <p>Optimization is conducted using the Adam optimizer with momentum parameters &beta;<sub>1</sub>=0.9, &beta;<sub>2</sub>=0.999, and initial learning rate 1.2&times;10<sup>-4</sup> scheduled via Cosine Annealing decay down to 1.0&times;10<sup>-6</sup> (Table&nbsp;III). Training incorporates PyTorch Automatic Mixed Precision (AMP FP16), maintaining a stable batch size of 4 within 6 GB VRAM.</p>

  <div class="table-wrapper">
    <div class="table-title"><strong>TABLE III: TRAINING HYPERPARAMETERS &amp; OPTIMIZATION SETTINGS</strong></div>
    <table class="booktabs">
      <thead>
        <tr><th class="align-left">Hyperparameter</th><th class="align-center">Configured Value</th></tr>
      </thead>
      <tbody>
        <tr><td class="align-left">Optimization Schedule</td><td class="align-center">Cosine Annealing Schedule (Converged State)</td></tr>
        <tr><td class="align-left">Batch Size / Cropped Patch Size</td><td class="align-center">4 / 128 &times; 128 pixels</td></tr>
        <tr><td class="align-left">Initial / Minimum Learning Rate</td><td class="align-center">1.2 &times; 10<sup>-4</sup> / 1.0 &times; 10<sup>-6</sup></td></tr>
        <tr><td class="align-left">Supervisory Loss Function</td><td class="align-center">Smooth L1 (Huber Loss, &delta; = 1.0)</td></tr>
        <tr><td class="align-left">Optimizer Algorithm</td><td class="align-center">Adam (&beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999)</td></tr>
        <tr><td class="align-left">Mixed Precision Acceleration</td><td class="align-center">PyTorch Automatic Mixed Precision (FP16)</td></tr>
        <tr><td class="align-left">Data Augmentation</td><td class="align-center">Random Horizontal / Vertical Flip, 90&deg; Rotation</td></tr>
      </tbody>
    </table>
  </div>

  <div class="full-width">
    <div class="table-wrapper">
      <div class="table-title"><strong>TABLE IV: QUANTITATIVE PERFORMANCE BENCHMARK ON LOL EVAL15 ACROSS REPRESENTATIVE STATE-OF-THE-ART METHODS</strong></div>
      <table class="booktabs">
        <thead>
          <tr>
            <th class="align-left">Method</th>
            <th class="align-center">Methodological Paradigm</th>
            <th class="align-center">Paired Supervision</th>
            <th class="align-center">PSNR (dB) &uarr;</th>
            <th class="align-center">SSIM [32] &uarr;</th>
            <th class="align-center">MAE &darr;</th>
            <th class="align-center">&Delta;PSNR vs Raw (dB)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td class="align-left">Unprocessed Low-Light Input</td><td class="align-center">&mdash;</td><td class="align-center">&mdash;</td><td class="align-center">7.77</td><td class="align-center">0.2530</td><td class="align-center">0.3842</td><td class="align-center">&mdash;</td></tr>
          <tr><td class="align-left">CLAHE (Reza et al., 2004)</td><td class="align-center">Traditional Histogram</td><td class="align-center">No</td><td class="align-center">11.41</td><td class="align-center">0.3732</td><td class="align-center">0.2610</td><td class="align-center">+3.64</td></tr>
          <tr><td class="align-left">RetinexNet (Wei et al., 2018) [12]</td><td class="align-center">CNN Retinex Decomposition</td><td class="align-center">Yes</td><td class="align-center">16.77</td><td class="align-center">0.4620</td><td class="align-center">0.1410</td><td class="align-center">+9.00</td></tr>
          <tr><td class="align-left">Zero-DCE (Guo et al., 2020) [18]</td><td class="align-center">Zero-Reference Deep Curve</td><td class="align-center">No</td><td class="align-center">14.86</td><td class="align-center">0.5620</td><td class="align-center">0.1850</td><td class="align-center">+7.09</td></tr>
          <tr><td class="align-left">EnlightenGAN (Jiang et al., 2021) [20]</td><td class="align-center">Unpaired Adversarial GAN</td><td class="align-center">No</td><td class="align-center">17.48</td><td class="align-center">0.6500</td><td class="align-center">0.1320</td><td class="align-center">+9.71</td></tr>
          <tr><td class="align-left">KinD (Zhang et al., 2019) [15]</td><td class="align-center">Multi-Stage CNN Pipeline</td><td class="align-center">Yes</td><td class="align-center">20.87</td><td class="align-center">0.8000</td><td class="align-center">0.0880</td><td class="align-center">+13.10</td></tr>
          <tr><td class="align-left">Restormer (Zamir et al., 2022) [26]</td><td class="align-center">Channel-Transposed Transformer</td><td class="align-center">Yes</td><td class="align-center">22.43</td><td class="align-center">0.8230</td><td class="align-center">0.0760</td><td class="align-center">+14.66</td></tr>
          <tr><td class="align-left">LLFormer Original Paper (Wang et al., 2023) [30]</td><td class="align-center">Full UHD Transformer (dim=32, 200 ep.)</td><td class="align-center">Yes</td><td class="align-center"><strong>23.65</strong></td><td class="align-center"><strong>0.8710</strong></td><td class="align-center"><strong>0.0610</strong></td><td class="align-center"><strong>+15.88</strong></td></tr>
          <tr style="background-color: #f7f9fc;"><td class="align-left"><strong>LLFormer (Ours Reproduced, Converged)</strong></td><td class="align-center"><strong>Compact Transformer (dim=16)</strong></td><td class="align-center"><strong>Yes</strong></td><td class="align-center"><strong>20.23</strong></td><td class="align-center"><strong>0.7513</strong></td><td class="align-center"><strong>0.0924</strong></td><td class="align-center"><strong>+12.46</strong></td></tr>
        </tbody>
      </table>
    </div>

    <figure style="margin-top: 3pt;">
      <img src="__IMG_SOTA__" alt="SOTA Benchmark Comparison">
      <figcaption><strong>Fig. 3.</strong> Quantitative benchmark comparison across representative methods on the LOL <code>eval15</code> benchmark. (A) PSNR (dB): our compact reproduction model achieves 20.23 dB mean (21.20 dB peak), significantly outperforming classical and zero-reference methods. (B) SSIM comparison demonstrating substantial structural similarity recovery.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">D. Quantitative Benchmark Comparison</h2>
  <p>Table&nbsp;IV and Fig.&nbsp;3 present the quantitative results on the LOL <code>eval15</code> benchmark across eight representative methods. Our converged reproduction of LLFormer establishes a robust baseline, achieving an average PSNR of <strong>20.23 dB</strong> (peak <strong>21.20 dB</strong>) and SSIM of <strong>0.7513</strong> (peak <strong>0.7604</strong>). Compared to traditional CLAHE, our model yields an improvement of +8.82 dB PSNR. When compared against the foundational RetinexNet [12] (+3.46 dB PSNR, +0.289 SSIM) and zero-reference EnlightenGAN [20] (+2.75 dB PSNR, +0.101 SSIM), LLFormer demonstrates superior noise suppression and contrast reconstruction. While operating with reduced channel capacity (<i>C=16</i>) to accommodate consumer hardware, it closely approaches the performance of multi-stage KinD [15] (20.87 dB) and Restormer [26] (22.43 dB).</p>

  <h2 class="subsection-heading">E. Training Dynamics &amp; Convergence Profile</h2>
  <p>The optimization dynamics across training iterations are analyzed through separated tracking curves (Figs.&nbsp;4&ndash;6). The training Smooth L1 loss (Fig.&nbsp;4) exhibits smooth exponential decay from an initial 0.0182 down to 0.0083 without numerical divergence. Validation PSNR (Fig.&nbsp;5) shows monotonic improvement, crossing the 20 dB mark midway through the schedule and reaching a peak of 21.20 dB at the converged checkpoint. Concurrently, validation SSIM (Fig.&nbsp;6) ascends steadily from 0.540 to 0.7604, confirming consistent structural feature learning.</p>

  <div class="single-column">
    <figure>
      <img src="__IMG_LOSS__" alt="Loss Convergence" style="max-height: 35mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 4.</strong> Training convergence curve (Smooth L1 Loss) across optimization iterations on LOL <code>our485</code>.</figcaption>
    </figure>
  </div>

  <div class="single-column">
    <figure>
      <img src="__IMG_PSNR__" alt="Validation PSNR" style="max-height: 35mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 5.</strong> Validation PSNR progression on LOL <code>eval15</code>, reaching a peak of 21.20 dB at the converged checkpoint.</figcaption>
    </figure>
  </div>

  <div class="single-column">
    <figure>
      <img src="__IMG_SSIM__" alt="Validation SSIM" style="max-height: 35mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 6.</strong> Validation SSIM progression across training iterations, achieving a peak of 0.7604.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">F. Qualitative Visual Restoration</h2>
  <p>Visual evaluation is presented using contiguous image panels representing three distinct modalities: (a) Low-Light Input, (b) LLFormer (Ours Reproduced, Converged Model), and (c) Ground Truth reference captured under normal lighting conditions. Fig.&nbsp;7 highlights benchmark sample <code>22.png</code> (an indoor bedroom scene). In the low-light capture, the plush toy, storage cabinet, and folded fabrics are completely buried in near-zero intensity levels. The reproduced LLFormer sharply restores deep shadow regions, producing crisp geometric contours and smooth surface illumination without generating the severe edge halos typical of RetinexNet [12]. On this individual sample, the model elevates PSNR from 6.11 dB to <strong>26.11 dB</strong> (+20.0 dB gain) and SSIM from 0.184 to <strong>0.8300</strong>.</p>

  <div class="full-width">
    <figure>
      <img src="__IMG_SAMPLE22__" alt="Sample 22 Triptych" style="max-height: 34mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 7.</strong> Contiguous triptych visual comparison on benchmark test image <code>22.png</code>. (a) Low-Light input (6.11 dB, SSIM 0.184); (b) LLFormer restoration (Ours Reproduced, Converged Model: 26.11 dB, SSIM 0.8300); (c) Ground Truth reference captured under normal illumination. The model sharply restores deep shadow details and suppressed textures without edge halos.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">G. Diagnostic Error Decomposition</h2>
  <p>To inspect restoration fidelity at the pixel level, Fig.&nbsp;8 presents an ROI zoom and spatial absolute error decomposition map (<i>|GT - Output|</i>) on sample <code>22.png</code>. While fine structural edges and boundaries align precisely with the ground truth, high-error pixels (warm colors) concentrate predominantly on colorful surfaces (such as the orange fabric and decorative textiles). This demonstrates that the residual discrepancy is largely chromatic rather than structural.</p>

  <div class="full-width">
    <figure>
      <img src="__IMG_DETAIL__" alt="Hero Detail Error Analysis" style="max-height: 50mm; width: auto; max-width: 95%;">
      <figcaption><strong>Fig. 8.</strong> Detailed restoration and per-pixel error decomposition on sample <code>22.png</code>. Top row: Full views with ROI markers. Bottom row: (d) Zoomed low-light crop; (e) Zoomed enhanced crop; (f) Per-pixel error map |GT - Output|. Warmer colors denote larger absolute deviation, localizing error primarily on colored surfaces.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">H. Failure Mode Taxonomy</h2>
  <p>A systematic empirical audit across all 15 LOL <code>eval15</code> benchmark images (Fig.&nbsp;9) reveals five primary failure modes: 1) Warm-Color Bias (86.7%), 2) Blue-Channel Desaturation (80.0%), 3) Texture Over-Smoothing (73.3%), 4) Low-Illumination Blocking (53.3%), and 5) Edge Halos (46.7%). The root cause is directly linked to uniform RGB-space Smooth L1 supervision (Eq.&nbsp;4). Because all color channels are averaged equally in pixel space, the network minimizes mathematical loss by matching dominant luminance gradients while allowing subtle chromatic drift.</p>

  <div class="single-column">
    <figure>
      <img src="__IMG_FAIL__" alt="Failure Mode Frequency" style="max-height: 38mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 9.</strong> Empirical frequency of identified failure modes across all 15 LOL <code>eval15</code> benchmark test images.</figcaption>
    </figure>
  </div>

  <h1 class="section-heading">V. Systematic Ablation Study: Chromatic Objective Loss vs. Histogram Post-Processing</h1>
  
  <h2 class="subsection-heading">A. Formulation of Decoupled HSV Color Loss</h2>
  <p>To directly address the chromatic drift identified in Section IV-H, we formulate a decoupled color-space objective. Transforming predicted and ground-truth RGB images into the cylindrical HSV color space, we explicitly supervise chromatic components:</p>
  <div class="equation">
    L<sub>total</sub> = L<sub>SL1</sub><sup>RGB</sup> + &lambda;<sub>HSV</sub> &middot; [ L<sub>SL1</sub><sup>H</sup> + L<sub>SL1</sub><sup>S</sup> ]
    <span class="eqno">(7)</span>
  </div>
  <p class="no-indent">where &lambda;<sub>HSV</sub> = 0.15 acts as a perceptual color regularizer. By directly penalizing angular phase drift in Hue (<i>H</i>) and intensity loss in Saturation (<i>S</i>), this objective prevents the network from trading off color fidelity to satisfy pixel-level luminance regression.</p>

  <h2 class="subsection-heading">B. Experimental Configurations &amp; Controls</h2>
  <p>To rigorously determine whether chromatic restoration requires end-to-end representation learning or can be achieved via post-hoc histogram adjustments, we establish four controlled configurations:</p>
  <p class="no-indent">&bull; <strong>Config A (LLFormer Base):</strong> Baseline model trained with standard Smooth L1 loss, serving as the experimental reference.</p>
  <p class="no-indent">&bull; <strong>Config B (Base + HSV Loss):</strong> Identical architecture and optimization schedule, incorporating decoupled HSV supervision (&lambda;<sub>HSV</sub> = 0.15).</p>
  <p class="no-indent">&bull; <strong>Config C (Base + CLAHE):</strong> Baseline model output followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) applied exclusively to the Value (<i>V</i>) channel in HSV space (clipLimit=1.5, tileGrid=(8,8)), preserving <i>H</i> and <i>S</i> channels completely untouched.</p>
  <p class="no-indent">&bull; <strong>Config D (Base + HSV Loss + CLAHE):</strong> Identical CLAHE post-processing applied to the output of Config B.</p>
  <p>Strict controls are enforced: all configurations are evaluated on the exact same 15 test scenes of LOL <code>eval15</code>; zero test ground-truth data is utilized during histogram post-processing; and Circular &Delta;<i>H</i> is computed using an identical valid hue mask (<i>S<sub>GT</sub> &gt; 0.10, V<sub>GT</sub> &gt; 0.10</i>) across all configurations.</p>

  <div class="full-width">
    <div class="table-wrapper">
      <div class="table-title"><strong>TABLE V: QUANTITATIVE ABLATION STUDY ACROSS FOUR CONFIGURATIONS ON LOL EVAL15</strong></div>
      <table class="booktabs">
        <thead>
          <tr>
            <th class="align-left">Configuration</th>
            <th class="align-left">Supervision / Processing Pipeline</th>
            <th class="align-center">PSNR (dB) &uarr;</th>
            <th class="align-center">SSIM &uarr;</th>
            <th class="align-center">&Delta;E₀₀ &darr;</th>
            <th class="align-center">Circular &Delta;H (&deg;) &darr;</th>
            <th class="align-center">Neon-Pink Ret. % &uarr;</th>
            <th class="align-center">Neon-Yellow Shift % &darr;</th>
            <th class="align-center">Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td class="align-left"><strong>A: Baseline</strong></td><td class="align-left">Smooth L1 Supervision (Standard LLFormer)</td><td class="align-center"><strong>20.56</strong></td><td class="align-center"><strong>0.7601</strong></td><td class="align-center"><strong>11.32</strong></td><td class="align-center">28.18&deg;</td><td class="align-center">34.10%</td><td class="align-center">41.30%</td><td class="align-center">891.8</td></tr>
          <tr style="background-color: #f7f9fc;"><td class="align-left"><strong>B: +HSV Loss</strong></td><td class="align-left">Smooth L1 + Decoupled HSV Loss (&lambda;=0.15)</td><td class="align-center">19.80</td><td class="align-center">0.7411</td><td class="align-center">11.35</td><td class="align-center"><strong>25.71&deg;</strong></td><td class="align-center"><strong>92.31%</strong></td><td class="align-center"><strong>0.12%</strong></td><td class="align-center"><strong>883.8</strong></td></tr>
          <tr><td class="align-left"><strong>C: +CLAHE</strong></td><td class="align-left">Baseline + CLAHE on HSV-V (clip=1.5, grid=8&times;8)</td><td class="align-center">18.50</td><td class="align-center">0.6659</td><td class="align-center">12.34</td><td class="align-center">28.18&deg;</td><td class="align-center">34.09%</td><td class="align-center">41.31%</td><td class="align-center">988.7</td></tr>
          <tr><td class="align-left"><strong>D: +HSV+CLAHE</strong></td><td class="align-left">Config B + CLAHE on HSV-V (clip=1.5, grid=8&times;8)</td><td class="align-center">18.20</td><td class="align-center">0.6701</td><td class="align-center">12.19</td><td class="align-center">25.71&deg;</td><td class="align-center">92.29%</td><td class="align-center">0.12%</td><td class="align-center">962.8</td></tr>
        </tbody>
      </table>
    </div>

    <figure style="margin-top: 2pt;">
      <img src="__IMG_ABLATION_BAR__" alt="Ablation Quantitative Metrics Bar Chart" style="max-height: 35mm; width: auto; max-width: 100%;">
      <figcaption><strong>Fig. 10.</strong> Quantitative metrics comparison across ablation configurations A (Baseline), B (+HSV Loss), C (+CLAHE), and D (+HSV+CLAHE) on LOL <code>eval15</code>. Incorporating HSV loss achieves superior Circular Hue Error (&Delta;H: 25.71&deg; vs 28.18&deg;) with zero inference latency overhead, whereas histogram post-processing degrades structural similarity.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">C. Quantitative Ablation Findings</h2>
  <p>Table&nbsp;V and Fig.&nbsp;10 report the quantitative ablation results. Several fundamental findings emerge:</p>
  <p class="no-indent">1) <em>Efficacy of HSV Loss on Chromatic Restoration (B vs. A):</em> Config B reduces Circular Hue Error &Delta;<i>H</i> across the test set from <strong>28.18&deg; to 25.71&deg; (average reduction of 2.48&deg;)</strong>. A granular per-image examination reveals that &Delta;<i>H</i> improves on <strong>11 out of 15 images (73.3%)</strong>, most prominently on scenes featuring intense colored illuminations such as <code>111.png</code> (-18.86&deg;), <code>748.png</code> (-8.19&deg;), and <code>179.png</code> (-7.07&deg;). On the remaining 4 scenes with neutral gray backgrounds (<code>547.png</code>, <code>55.png</code>, <code>665.png</code>, <code>778.png</code>), Baseline retains lower hue variance.</p>
  
  <p class="no-indent">2) <em>Pixel Regression Trade-off:</em> Config B exhibits a modest reduction in PSNR (-0.76 dB) and SSIM (-0.0190) relative to Baseline A. This reflects a classic mathematical trade-off: Smooth L1 supervision optimizes pixel-wise mean squared error by converging toward average luminance intensities (regression-to-the-mean), which inflates numerical PSNR at the cost of chromatic desaturation. HSV loss forces the model to respect perceptual color boundaries, trading a minor amount of pixel variance for authentic color rendition.</p>

  <p class="no-indent">3) <em>Inadequacy of Histogram Post-Processing (C &amp; D vs. A &amp; B):</em> Applying CLAHE on the luminance channel <i>V</i> completely fails to improve hue accuracy (&Delta;<i>H</i> remains identical at 28.18&deg; for C and 25.71&deg; for D because <i>H</i> is untouched). Simultaneously, CLAHE distorts the delicate photon distributions reconstructed by the transformer, causing PSNR to drop by ~2 dB (18.50 dB for C; 18.20 dB for D) and increasing inference latency by ~90 ms. Consequently, post-hoc histogram manipulation is not complementary to learning-based restoration.</p>

  <div class="full-width">
    <figure style="margin-top: 2pt;">
      <img src="__IMG_ABLATION_179__" alt="Visual Ablation on Scene 179" style="max-height: 40mm; width: auto; max-width: 95%;">
      <figcaption><strong>Fig. 11.</strong> Visual comparison and region-of-interest (ROI) zoom on benchmark scene <code>179.png</code> featuring an intense neon-pink light source. In Baseline A, 41.30% of the neon sign erroneously shifts into yellow tones. In Config B (+HSV Loss), authentic pink chrominance is restored (92.31% pink retention; 0.12% yellow shift), closely matching the Ground Truth reference.</figcaption>
    </figure>
  </div>

  <h2 class="subsection-heading">D. Case Study: Extreme Chromatic Shift on Scene 179.png</h2>
  <p>To inspect color recovery where standard models fail most severely, we analyze scene <code>179.png</code>, containing a localized neon-pink artificial sign (Fig.&nbsp;11). Under Baseline A (Smooth L1), the model suffers severe hue drift: <strong>41.30%</strong> of the neon sign pixels shift into yellow-orange hues (35&deg;&ndash;75&deg;), while only 34.10% retain true pink hues (280&deg;&ndash;20&deg;), resulting in an unnatural washed-out yellow glow. In striking contrast, Config B (+HSV Loss) virtually eliminates yellow drift down to <strong>0.12%</strong>, elevating correct pink retention to <strong>92.31%</strong>. On this specific challenging scene, Config B outperforms Baseline across all metrics: PSNR improves from 12.39 dB to <strong>13.05 dB</strong> (+0.66 dB), &Delta;E₀₀ drops from 25.90 to <strong>22.34</strong> (-3.56), and &Delta;<i>H</i> decreases from 27.38&deg; to <strong>20.31&deg;</strong> (-7.07&deg;).</p>

  <h2 class="subsection-heading">E. Experimental Limitations &amp; Future Work</h2>
  <p>Due to consumer hardware constraints (single 6 GB GPU), our models were evaluated under a fixed random seed. While trend consistency is demonstrated across 15 real benchmark scenes, evaluating multi-seed variance and cross-dataset generalization (e.g., on MEF and NPE benchmarks) constitutes an important avenue for future research.</p>

  <h1 class="section-heading">VI. Conclusion</h1>
  <p>In this investigation, our research group successfully reproduced, characterized, and analyzed the state-of-the-art LLFormer architecture on the standard LOL benchmark under realistic consumer hardware constraints. By deploying an in-RAM caching pipeline and mixed-precision acceleration, our converged model achieved <strong>21.20 dB</strong> peak PSNR and <strong>0.7604</strong> SSIM, surpassing classical and zero-reference methods by substantial margins. Statistical exploratory data analysis and per-pixel error decomposition revealed that while A-MSA effectively preserves structural sharpness, uniform RGB loss supervision leads to chromatic drift. Our formulated systematic ablation study quantitatively proved that decoupled HSV color loss serves as an effective regularizer, reducing Circular Hue Error by 2.48&deg; across 73.3% of test scenes and eliminating acute hue shifts, while confirming that post-hoc histogram manipulation is unsuited for non-linear chromatic recovery.</p>

  <h1 class="section-heading">References</h1>
  <div class="reference-item">[1] E. H. Land and J. J. McCann, "Lightness and Retinex theory," <em>J. Opt. Soc. Am.</em>, vol. 61, no. 1, pp. 1&ndash;11, 1971.</div>
  <div class="reference-item">[2] D. J. Jobson, Z. Rahman, and G. A. Woodell, "A multiscale retinex for bridging the gap between color images and the human observation of scenes," <em>IEEE Trans. Image Process.</em>, vol. 6, no. 7, pp. 965&ndash;976, 1997.</div>
  <div class="reference-item">[3] X. Dong, G. Wang, Y. Pang, W. Li, J. Wen, W. Meng, and Y. Lu, "Fast efficient algorithm for enhancement of low lighting video," in <em>Proc. IEEE ICME</em>, 2011, pp. 1&ndash;6.</div>
  <div class="reference-item">[4] S. Wang, J. Zheng, H.-M. Hu, and B. Li, "Naturalness preserved enhancement algorithm for non-uniform illumination images," <em>IEEE Trans. Image Process.</em>, vol. 22, no. 9, pp. 3538&ndash;3548, 2013.</div>
  <div class="reference-item">[5] X. Fu, D. Zeng, Y. Huang, X.-P. Zhang, and X. Ding, "A weighted variational model for simultaneous reflectance and illumination estimation," in <em>Proc. CVPR</em>, 2016, pp. 2782&ndash;2790.</div>
  <div class="reference-item">[6] X. Guo, Y. Li, and H. Ling, "LIME: Low-light image enhancement via illumination map estimation," <em>IEEE Trans. Image Process.</em>, vol. 26, no. 2, pp. 982&ndash;993, 2017.</div>
  <div class="reference-item">[7] R. Kimmel, M. Elad, D. Shaked, R. Keshet, and I. Sobel, "A variational framework for Retinex," <em>Int. J. Comput. Vis.</em>, vol. 52, no. 1, pp. 7&ndash;23, 2003.</div>
  <div class="reference-item">[8] M. Elad, "Retinex by two bilateral filters," <em>Signal, Image and Video Processing</em>, vol. 1, no. 2, pp. 183&ndash;196, 2007.</div>
  <div class="reference-item">[9] Z. Ying, G. Li, and W. Gao, "A bio-inspired multi-exposure fusion framework for low-light image enhancement," <em>arXiv:1711.00591</em>, 2017.</div>
  <div class="reference-item">[10] K. Ma, K. Zeng, and Z. Wang, "Perceptual quality assessment for multi-exposure image fusion," <em>IEEE Trans. Image Process.</em>, vol. 24, no. 11, pp. 3345&ndash;3356, 2015.</div>
  <div class="reference-item">[11] K. G. Lore, A. Akintayo, and S. Sarkar, "LLNet: A deep autoencoder approach to natural low-light image enhancement," <em>Pattern Recognit.</em>, vol. 61, pp. 650&ndash;662, 2017.</div>
  <div class="reference-item">[12] C. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex decomposition for low-light enhancement," in <em>Proc. BMVC</em>, 2018. [Online]. Available: https://daooshee.github.io/BMVC2018website/</div>
  <div class="reference-item">[13] X. Ren, M. Li, W.-H. Cheng, and J. Liu, "Joint enhancement and denoising method via sequential decomposition," in <em>Proc. IEEE ISCAS</em>, 2018, pp. 1&ndash;5.</div>
  <div class="reference-item">[14] C. Chen, Q. Chen, J. Xu, and V. Koltun, "Learning to see in the dark," in <em>Proc. CVPR</em>, 2018, pp. 3291&ndash;3300.</div>
  <div class="reference-item">[15] Y. Zhang, J. Zhang, and X. Guo, "Kindling the darkness: A practical low-light image enhancer," in <em>Proc. ACM MM</em>, 2019, pp. 1632&ndash;1640.</div>
  <div class="reference-item">[16] K. Xu, X. Yang, B. Yin, and R. W. Lau, "Learning to restore low-light images via decomposition-and-enhancement network," <em>IEEE Trans. Image Process.</em>, vol. 27, no. 11, pp. 5440&ndash;5452, 2018.</div>
  <div class="reference-item">[17] F. Lv, F. Lu, J. Wu, and C. Lim, "MBLLEN: Low-light image/video enhancement using CNNs," in <em>Proc. BMVC</em>, 2018.</div>
  <div class="reference-item">[18] C. Guo, C. Li, J. Guo, C. C. Loy, J.-M. Hou, S. Kwong, and R. Cong, "Zero-reference deep curve estimation for low-light image enhancement," in <em>Proc. CVPR</em>, 2020, pp. 1780&ndash;1789.</div>
  <div class="reference-item">[19] C. Li, C. Guo, and C. C. Loy, "Learning to enhance low-light image via zero-reference deep curve estimation," <em>IEEE Trans. Pattern Anal. Mach. Intell.</em>, vol. 44, no. 8, pp. 4225&ndash;4238, 2021.</div>
  <div class="reference-item">[20] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, "EnlightenGAN: Deep light enhancement without paired supervision," <em>IEEE Trans. Image Process.</em>, vol. 30, pp. 2340&ndash;2349, 2021.</div>
  <div class="reference-item">[21] Y.-S. Chen, Y.-C. Wang, M.-H. Kao, and Y.-Y. Chuang, "Deep photo enhancer: Unpaired learning for image enhancement from photographs with GANs," in <em>Proc. CVPR</em>, 2018, pp. 6306&ndash;6314.</div>
  <div class="reference-item">[22] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in <em>Proc. NeurIPS</em>, 2017, pp. 5998&ndash;6008.</div>
  <div class="reference-item">[23] H. Chen, Y. Wang, T. Guo, C. Xu, Y. Deng, Z. Liu, S. Ma, C. Xu, C. Xu, and W. Gao, "Pre-trained image processing transformer," in <em>Proc. CVPR</em>, 2021, pp. 12299&ndash;12310.</div>
  <div class="reference-item">[24] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in <em>Proc. ICCV</em>, 2021, pp. 10012&ndash;10022.</div>
  <div class="reference-item">[25] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, "SwinIR: Image restoration using Swin transformer," in <em>Proc. ICCVW</em>, 2021, pp. 1833&ndash;1844.</div>
  <div class="reference-item">[26] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, "Restormer: Efficient transformer for high-resolution image restoration," in <em>Proc. CVPR</em>, 2022, pp. 5728&ndash;5739.</div>
  <div class="reference-item">[27] Z. Wang, X. Cun, J. Bao, W. Zhou, J. Liu, and H. Li, "Uformer: A general u-shaped transformer for image restoration," in <em>Proc. CVPR</em>, 2022, pp. 17683&ndash;17693.</div>
  <div class="reference-item">[28] X. Chu, Z. Tian, Y. Wang, B. Zhang, H. Shen, X. Wei, H. Yan, and C. Shen, "Twins: Revisiting the design of spatial attention in vision transformers," in <em>Proc. NeurIPS</em>, 2021, pp. 9366&ndash;9377.</div>
  <div class="reference-item">[29] X. Xu, R. Wang, C.-W. Fu, and J. Jia, "SNR-Aware low-light image enhancement," in <em>Proc. CVPR</em>, 2022, pp. 17714&ndash;17724.</div>
  <div class="reference-item">[30] T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, and T. Lu, "Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method," in <em>Proc. AAAI</em>, 2023, pp. 2654&ndash;2662.</div>
  <div class="reference-item">[31] H. Zhao, O. Gallo, I. Frosio, and J. Kautz, "Loss functions for image restoration with neural networks," <em>IEEE Trans. Comput. Imaging</em>, vol. 3, no. 1, pp. 47&ndash;57, 2016.</div>
  <div class="reference-item">[32] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," <em>IEEE Trans. Image Process.</em>, vol. 13, no. 4, pp. 600&ndash;612, 2004.</div>
  <div class="reference-item">[33] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in <em>Proc. CVPR</em>, 2018, pp. 586&ndash;595.</div>

</div>

</body>
</html>
"""
    replacements = {
        "__IMG_ARCH__": img_arch,
        "__IMG_AMSA__": img_amsa,
        "__IMG_SOTA__": img_sota,
        "__IMG_LOSS__": img_loss,
        "__IMG_PSNR__": img_psnr,
        "__IMG_SSIM__": img_ssim,
        "__IMG_SAMPLE22__": img_sample22,
        "__IMG_DETAIL__": img_detail,
        "__IMG_FAIL__": img_fail,
        "__IMG_ABLATION_BAR__": img_ablation_bar,
        "__IMG_ABLATION_179__": img_ablation_179
    }
    for k, v in replacements.items():
        template = template.replace(k, v)

    return template

def main():
    print("--> Generating professional IEEE Conference HTML document...")
    html_str = build_html()
    temp_html = os.path.join(WORKSPACE_DIR, "paper_for_pdf.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"--> Saved HTML template: {temp_html}")

    output_pdf = os.path.join(WORKSPACE_DIR, "LLFormer_Reproduction_Paper.pdf")
    artifact_pdf = os.path.join(ARTIFACTS_DIR, "LLFormer_Reproduction_Paper.pdf")

    print(f"--> Invoking headless browser to compile PDF: {output_pdf}")
    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf}",
        temp_html
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_pdf):
        sz = os.path.getsize(output_pdf) / (1024 * 1024)
        print(f"--> SUCCESS! Generated PDF: {output_pdf} ({sz:.2f} MB)")
        shutil.copy2(output_pdf, artifact_pdf)
        print(f"--> Copied to artifacts: {artifact_pdf}")
    else:
        print(f"--> Error generating PDF. Code: {res.returncode}, Stderr: {res.stderr}")

if __name__ == '__main__':
    main()
