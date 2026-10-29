# 数据管理
原始数据仅留本机 data/raw/，预处理产物放 data/processed/。
暂定 Middlebury Stereo v3、NYU Depth V2 及自录静态/移动短视频；需核对技术 PDF 原件。
下载时补记官方 URL、版本、许可、校验值、预处理步骤；不擅自重新分发数据。
data/manifests/ 只提交样本 ID、相对路径、划分等小型元数据。两人使用相同划分；视频避免相邻片段跨训练/验证/测试泄漏。
自录视频记录场景、分辨率、fps、时长、相机信息及隐私许可。
