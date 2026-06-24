# Results

This page summarizes the quantitative results reported in the current paper
draft. All external benchmark comparisons are target-training-free unless noted
otherwise.

## Average Primary AP on 9 External Benchmarks

<div align="center">
<table style="min-width: 70%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th style="text-align: center; padding: 12px 20px">Average Primary AP</th>
      <th style="text-align: center; padding: 12px 20px">Delta vs LEVIRDetNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DynamicVis-L</td>
      <td style="text-align: center; padding: 10px 20px">70.52</td>
      <td style="text-align: center; padding: 10px 20px">+10.04</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">YOLOv12x</td>
      <td style="text-align: center; padding: 10px 20px">75.17</td>
      <td style="text-align: center; padding: 10px 20px">+5.39</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DEIMv2 (DINOv3)</td>
      <td style="text-align: center; padding: 10px 20px">75.54</td>
      <td style="text-align: center; padding: 10px 20px">+5.02</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px"><strong>LEVIRDetNet</strong></td>
      <td style="text-align: center; padding: 10px 20px"><strong>80.56</strong></td>
      <td style="text-align: center; padding: 10px 20px"><strong>-</strong></td>
    </tr>
  </tbody>
</table>
</div>

## Per-Benchmark Primary Metric Breakdown

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th style="border-right: 2px solid #ddd; padding: 12px 20px">Dataset</th>
      <th style="text-align: center; padding: 12px 20px">Primary Metric</th>
      <th style="text-align: center; padding: 12px 20px">Best Prior Method</th>
      <th style="text-align: center; padding: 12px 20px">Best Prior</th>
      <th style="text-align: center; padding: 12px 20px">LEVIRDetNet</th>
      <th style="text-align: center; padding: 12px 20px">Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">ADCOS</td><td style="text-align: center; padding: 10px 20px">AP<sub>bbox</sub></td><td style="text-align: center; padding: 10px 20px">RTMDet</td><td style="text-align: center; padding: 10px 20px">79.45</td><td style="text-align: center; padding: 10px 20px"><strong>83.60</strong></td><td style="text-align: center; padding: 10px 20px">+4.15</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">UCAS-AOD</td><td style="text-align: center; padding: 10px 20px">AP<sub>bbox</sub></td><td style="text-align: center; padding: 10px 20px">DEIMv2 (DINOv3+ViT-Base)</td><td style="text-align: center; padding: 10px 20px">75.38</td><td style="text-align: center; padding: 10px 20px"><strong>80.40</strong></td><td style="text-align: center; padding: 10px 20px">+5.02</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">HRPlane-v2</td><td style="text-align: center; padding: 10px 20px">AP<sub>bbox</sub></td><td style="text-align: center; padding: 10px 20px">DEIMv2 (DINOv3+ViT-Base)</td><td style="text-align: center; padding: 10px 20px">78.87</td><td style="text-align: center; padding: 10px 20px"><strong>81.92</strong></td><td style="text-align: center; padding: 10px 20px">+3.05</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">CORS-ADD</td><td style="text-align: center; padding: 10px 20px">AP<sub>bbox</sub></td><td style="text-align: center; padding: 10px 20px">YOLOv12x</td><td style="text-align: center; padding: 10px 20px">71.70</td><td style="text-align: center; padding: 10px 20px"><strong>72.06</strong></td><td style="text-align: center; padding: 10px 20px">+0.36</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">SkyFusion-plane</td><td style="text-align: center; padding: 10px 20px">AP<sub>50</sub></td><td style="text-align: center; padding: 10px 20px">DEIMv2 (DINOv3+ViT-Base)</td><td style="text-align: center; padding: 10px 20px">97.58</td><td style="text-align: center; padding: 10px 20px"><strong>98.27</strong></td><td style="text-align: center; padding: 10px 20px">+0.69</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">VHRV</td><td style="text-align: center; padding: 10px 20px">AP<sub>bbox</sub></td><td style="text-align: center; padding: 10px 20px">YOLOv12x</td><td style="text-align: center; padding: 10px 20px">72.58</td><td style="text-align: center; padding: 10px 20px"><strong>73.55</strong></td><td style="text-align: center; padding: 10px 20px">+0.97</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">SkyFusion-ship</td><td style="text-align: center; padding: 10px 20px">AP<sub>50</sub></td><td style="text-align: center; padding: 10px 20px">YOLOv5x</td><td style="text-align: center; padding: 10px 20px">47.70</td><td style="text-align: center; padding: 10px 20px"><strong>60.39</strong></td><td style="text-align: center; padding: 10px 20px">+12.69</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">NWPU</td><td style="text-align: center; padding: 10px 20px">mAP</td><td style="text-align: center; padding: 10px 20px">DEIMv2 (DINOv3+ViT-Base)</td><td style="text-align: center; padding: 10px 20px">73.60</td><td style="text-align: center; padding: 10px 20px"><strong>76.04</strong></td><td style="text-align: center; padding: 10px 20px">+2.44</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">CarPK</td><td style="text-align: center; padding: 10px 20px">AP<sub>50</sub></td><td style="text-align: center; padding: 10px 20px">DSTDA</td><td style="text-align: center; padding: 10px 20px">98.20</td><td style="text-align: center; padding: 10px 20px"><strong>98.79</strong></td><td style="text-align: center; padding: 10px 20px">+0.59</td></tr>
  </tbody>
</table>
</div>

## Open-Set Threshold-Sweep Peak mAP

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th style="border-right: 2px solid #ddd; padding: 12px 20px">Dataset</th>
      <th style="text-align: center; padding: 12px 20px">LAE-DINO</th>
      <th style="text-align: center; padding: 12px 20px">SAM3</th>
      <th style="text-align: center; padding: 12px 20px">DEIMv2-DIOR</th>
      <th style="text-align: center; padding: 12px 20px">DEIMv2-xView</th>
      <th style="text-align: center; padding: 12px 20px">DEIMv2-DOTA</th>
      <th style="text-align: center; padding: 12px 20px">LEVIRDetNet</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">AOSD</td><td style="text-align: center; padding: 10px 20px">22.66</td><td style="text-align: center; padding: 10px 20px">15.48</td><td style="text-align: center; padding: 10px 20px">28.26</td><td style="text-align: center; padding: 10px 20px">11.82</td><td style="text-align: center; padding: 10px 20px">36.97</td><td style="text-align: center; padding: 10px 20px"><strong>54.63 (+17.65)</strong></td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">RarePlanes</td><td style="text-align: center; padding: 10px 20px">14.16</td><td style="text-align: center; padding: 10px 20px">36.92</td><td style="text-align: center; padding: 10px 20px">51.59</td><td style="text-align: center; padding: 10px 20px">29.62</td><td style="text-align: center; padding: 10px 20px">15.57</td><td style="text-align: center; padding: 10px 20px"><strong>59.93 (+8.34)</strong></td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">RSD-GOD</td><td style="text-align: center; padding: 10px 20px">40.25</td><td style="text-align: center; padding: 10px 20px">14.92</td><td style="text-align: center; padding: 10px 20px">25.51</td><td style="text-align: center; padding: 10px 20px">18.21</td><td style="text-align: center; padding: 10px 20px">27.71</td><td style="text-align: center; padding: 10px 20px"><strong>48.71 (+8.46)</strong></td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">SkyFusion-ship</td><td style="text-align: center; padding: 10px 20px">1.34</td><td style="text-align: center; padding: 10px 20px">1.48</td><td style="text-align: center; padding: 10px 20px">2.34</td><td style="text-align: center; padding: 10px 20px">4.27</td><td style="text-align: center; padding: 10px 20px">5.52</td><td style="text-align: center; padding: 10px 20px"><strong>27.63 (+22.11)</strong></td></tr>
  </tbody>
</table>
</div>

## Runtime and Performance

<div align="center">
<table style="min-width: 70%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th style="border-right: 2px solid #ddd; padding: 12px 20px">Method</th>
      <th style="text-align: center; padding: 12px 20px">FPS</th>
      <th style="text-align: center; padding: 12px 20px">Latency (ms)</th>
      <th style="text-align: center; padding: 12px 20px">FLOPs (G)</th>
      <th style="text-align: center; padding: 12px 20px">mAP<sub>close</sub></th>
      <th style="text-align: center; padding: 12px 20px">mAP<sub>open</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">LAE-DINO</td><td style="text-align: center; padding: 10px 20px">9.00</td><td style="text-align: center; padding: 10px 20px">111.10</td><td style="text-align: center; padding: 10px 20px">300</td><td style="text-align: center; padding: 10px 20px">-</td><td style="text-align: center; padding: 10px 20px">19.60</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM3</td><td style="text-align: center; padding: 10px 20px">3.86</td><td style="text-align: center; padding: 10px 20px">258.87</td><td style="text-align: center; padding: 10px 20px">5036.94</td><td style="text-align: center; padding: 10px 20px">-</td><td style="text-align: center; padding: 10px 20px">17.20</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">YOLOv12x</td><td style="text-align: center; padding: 10px 20px">96.30</td><td style="text-align: center; padding: 10px 20px">10.38</td><td style="text-align: center; padding: 10px 20px">184.6</td><td style="text-align: center; padding: 10px 20px">75.17</td><td style="text-align: center; padding: 10px 20px">-</td></tr>
    <tr><td style="border-right: 2px solid #ddd; padding: 10px 20px">DEIMv2</td><td style="text-align: center; padding: 10px 20px">18.55</td><td style="text-align: center; padding: 10px 20px">53.91</td><td style="text-align: center; padding: 10px 20px">174</td><td style="text-align: center; padding: 10px 20px">75.54</td><td style="text-align: center; padding: 10px 20px">26.93</td></tr>
    <tr style="border-top: 2px solid #b19c9cff"><td style="border-right: 2px solid #ddd; padding: 10px 20px"><strong>LEVIRDetNet</strong></td><td style="text-align: center; padding: 10px 20px"><strong>15.20</strong></td><td style="text-align: center; padding: 10px 20px"><strong>65.78</strong></td><td style="text-align: center; padding: 10px 20px"><strong>216.0</strong></td><td style="text-align: center; padding: 10px 20px"><strong>80.56</strong></td><td style="text-align: center; padding: 10px 20px"><strong>47.73</strong></td></tr>
  </tbody>
</table>
</div>

