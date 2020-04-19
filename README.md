# DistilBertSumm

**This code is entirely based on the work of @nlpyang (https://github.com/nlpyang/PreSumm) which is based on EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**

Presumm, the original project is based on building summarization models using pretrained encoders, BERT. I was interested in this work in a way of building lighter versions of those models based on distilbert.

My first experiment was to build an extractive summarization model half the size of the orginal BertSumExt. Despite the drastic reduction of size, I reached the same results as BertSumExt (losing 0.5 point at most).

My next experiments will be to build an abstractive summarizartion model and try to achieve state of the art results.


Results on CNN/DailyMail (20/8/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4">Extractive</td>
  </tr>
  <tr>
    <td class="tg-0pky">TransformerExt</td>
    <td class="tg-0pky">40.90</td>
    <td class="tg-0pky">18.02</td>
    <td class="tg-0pky">37.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt</td>
    <td class="tg-0pky">43.23</td>
    <td class="tg-0pky">20.24</td>
    <td class="tg-0pky">39.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (large)</td>
    <td class="tg-0pky">43.85</td>
    <td class="tg-0pky">20.34</td>
    <td class="tg-0pky">39.90</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="4">Abstractive</td>
  </tr>
  <tr>
    <td class="tg-0lax">TransformerAbs</td>
    <td class="tg-0lax">40.21</td>
    <td class="tg-0lax">17.76</td>
    <td class="tg-0lax">37.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumAbs</td>
    <td class="tg-0lax">41.72</td>
    <td class="tg-0lax">19.39</td>
    <td class="tg-0lax">38.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumExtAbs</td>
    <td class="tg-0lax">42.13</td>
    <td class="tg-0lax">19.60</td>
    <td class="tg-0lax">39.18</td>
  </tr>
  <tr>
    <td class="tg-0lax"><b>DistilBertSumExt (mine)<b></td>
    <td class="tg-0lax">**42.74**</td>
    <td class="tg-0lax">**19.98**</td>
    <td class="tg-0lax">**39.22**</td>
  </tr>
</table>

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge



**Updates**: For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Trained Models
[CNN/DM BertExt](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[CNN/DM BertExtAbs](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

[CNN/DM TransformerAbs](https://drive.google.com/open?id=1yLCqT__ilQ3mf5YUUCw9-UToesX5Roxy)

[XSum BertExtAbs](https://drive.google.com/open?id=1H50fClyTkNprWJNh10HWdGEdDdQIkzsI)

