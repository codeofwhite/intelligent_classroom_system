<template>
  <div class="monitor-container">

    <div class="section model-selector">
      <h3>模型配置</h3>
      <div class="controls">
        <label>当前推理模型：</label>
        <select v-model="selectedModel" @change="changeModel">
          <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
        </select>
        <span v-if="switching" class="loading-text">正在加载模型...</span>
      </div>
    </div>

    <div class="section live-section">
      <h3>实时监控画面 (AI 模式)</h3>
      <div class="video-window">
        <img src="http://localhost:5000/video_feed" alt="视频流加载中..." class="live-stream" />
      </div>
      <div class="status-panel">
        <p>状态：<span style="color: green">● 正在实时分析</span></p>
      </div>
    </div>

    <hr class="divider" />

    <div class="section upload-section">
      <h3>离线视频分析 (上传文件)</h3>
      <div class="upload-card">
        <input type="file" ref="videoInput" @change="onFileChange" accept="video/*" />
        <button @click="startAnalysis" :disabled="loading" class="upload-btn">
          {{ loading ? 'AI 正在分析帧...' : '上传并分析视频' }}
        </button>
      </div>

      <div v-if="resultUrl" class="result-display">
        <h4>分析结果回放：</h4>
        <video :src="resultUrl" controls class="result-video"></video>
        <p class="hint">文件已同步存储至 Docker MinIO 容器</p>
      </div>
    </div>

    <div class="history-section">

      <h3>历史分析记录 <button @click="fetchVideoList" class="refresh-btn">刷新列表</button></h3>

      <div class="video-grid">
        <div v-for="video in videoList" :key="video.name" class="video-item">
          <p class="video-date">{{ video.time }}</p>
          <video :src="video.url" controls width="300"></video>
          <p class="video-name">{{ video.name }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      videoFile: null,
      loading: false,
      resultUrl: '',
      videoList: [],
      modelOptions: [],
      selectedModel: '',
      switching: false
    };
  },
  mounted() {
    this.fetchModels(); // 页面加载时获取模型列表
    this.fetchVideoList(); // 页面加载时自动获取一次
  },
  methods: {
    async fetchModels() {
      try {
        const res = await axios.get('http://localhost:5000/get_models');
        this.modelOptions = res.data.models;
        this.selectedModel = res.data.current;
      } catch (err) {
        console.error("无法获取模型列表");
      }
    },
    async changeModel() {
      this.switching = true;
      try {
        await axios.post('http://localhost:5000/switch_model', {
          model_name: this.selectedModel
        });
        alert("模型切换成功！实时流和分析将使用新模型。");
      } catch (err) {
        alert("模型切换失败：" + err.response.data.msg);
      } finally {
        this.switching = false;
      }
    },
    async fetchVideoList() {
      try {
        const res = await axios.get('http://localhost:5000/list_videos');
        this.videoList = res.data;
      } catch (err) {
        console.error("获取列表失败", err);
      }
    },
    onFileChange(e) {
      this.videoFile = e.target.files[0];
    },
    async startAnalysis() {
      if (!this.videoFile) return alert("请选择视频");

      this.loading = true;
      const formData = new FormData();
      formData.append('video', this.videoFile);

      try {
        // 注意：这里确保指向你的 Flask 后端地址
        const res = await axios.post('http://localhost:5000/upload_video', formData);
        this.resultUrl = res.data.video_url;
      } catch (err) {
        console.error(err);
        alert("分析失败，请检查后端和 MinIO 状态");
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>

<style scoped>
.monitor-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.section {
  margin-bottom: 30px;
  padding: 15px;
  border: 1px solid #eee;
  border-radius: 12px;
}

.model-selector { background: #f9f9f9; border-left: 5px solid #42b983; }
.controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.loading-text { color: #666; font-size: 13px; font-style: italic; }
select { padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc; }

.live-stream {
  width: 100%;
  border-radius: 8px;
  background: #000;
  min-height: 300px;
}

.result-video {
  width: 100%;
  border-radius: 8px;
  margin-top: 10px;
}

.divider {
  margin: 40px 0;
  border: 0;
  border-top: 1px dashed #ccc;
}

.upload-btn {
  margin-top: 10px;
  padding: 8px 16px;
  cursor: pointer;
}

.hint {
  font-size: 12px;
  color: #666;
}

.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
}
.video-item {
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 8px;
}
.video-date { font-size: 12px; color: #888; }
.refresh-btn { font-size: 14px; margin-left: 10px; cursor: pointer; }
</style>