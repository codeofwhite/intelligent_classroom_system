<template>
  <div class="video-analysis-page">
    <div class="header">
      <h2>🎓 课堂行为视频分析系统</h2>
      <p>上传视频进行行为分析，查看历史分析记录</p>
    </div>

    <!-- 上传分析 -->
    <div class="card">
      <h3>📂 上传视频进行分析</h3>
      <div class="upload-area">
        <input
          type="file"
          ref="videoInput"
          @change="onFileChange"
          accept="video/*"
        />
        <button @click="startAnalysis" :disabled="loading" class="btn-primary">
          {{ loading ? "AI 分析中..." : "开始上传并分析" }}
        </button>
      </div>

      <!-- 结果回放 -->
      <div v-if="resultUrl" class="result-section">
        <h4>✅ 分析完成</h4>
        <video :src="resultUrl" controls class="result-video"></video>
        <p class="tip">文件已自动保存至云端存储</p>
      </div>
    </div>

    <!-- 历史记录 -->
    <div class="card">
      <div class="history-header">
        <h3>📜 历史分析记录</h3>
        <button @click="fetchVideoList" class="btn-outline">刷新</button>
      </div>

      <div class="video-grid">
        <div v-for="video in videoList" :key="video.name" class="video-card">
          <p class="time">{{ video.time }}</p>
          <video :src="video.url" controls></video>
          <p class="name">{{ video.name }}</p>
        </div>
        <div v-if="videoList.length === 0" class="empty">
          暂无历史记录
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const videoInput = ref(null)
const videoFile = ref(null)
const loading = ref(false)
const resultUrl = ref('')
const videoList = ref([])

// 选择文件
function onFileChange(e) {
  videoFile.value = e.target.files[0]
}

// 上传并分析
async function startAnalysis() {
  if (!videoFile.value) {
    alert('请选择视频')
    return
  }

  loading.value = true
  const formData = new FormData()
  formData.append('video', videoFile.value)

  try {
    const res = await axios.post('http://localhost:5000/upload_video', formData)
    resultUrl.value = res.data.video_url
  } catch (err) {
    alert('分析失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}

// 获取历史列表
async function fetchVideoList() {
  try {
    const res = await axios.get('http://localhost:5000/list_videos')
    videoList.value = res.data
  } catch (err) {
    console.error('获取失败')
  }
}

onMounted(() => {
  fetchVideoList()
})
</script>

<style scoped>
.video-analysis-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 30px 20px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f7f9fc;
  min-height: 100vh;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}
.header h2 {
  font-size: 24px;
  color: #222;
  margin-bottom: 8px;
}
.header p {
  color: #666;
  font-size: 15px;
}

.card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 30px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.card h3 {
  margin-top: 0;
  font-size: 18px;
  color: #333;
  margin-bottom: 18px;
}

.upload-area {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

input[type="file"] {
  padding: 6px;
  font-size: 14px;
}

.btn-primary {
  background: #42b983;
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}
.btn-primary:disabled {
  background: #a0d9bc;
  cursor: not-allowed;
}

.result-section {
  margin-top: 24px;
}
.result-video {
  width: 100%;
  border-radius: 10px;
  background: #000;
}
.tip {
  font-size: 13px;
  color: #888;
  margin-top: 8px;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.btn-outline {
  border: 1px solid #42b983;
  color: #42b983;
  background: white;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin-top: 16px;
}

.video-card {
  border: 1px solid #eee;
  border-radius: 12px;
  padding: 12px;
}
.video-card video {
  width: 100%;
  border-radius: 8px;
  background: #000;
}
.video-card .time {
  font-size: 12px;
  color: #999;
  margin: 0 0 6px 0;
}
.video-card .name {
  font-size: 14px;
  color: #333;
  margin: 6px 0 0 0;
}

.empty {
  color: #999;
  grid-column: 1 / -1;
  text-align: center;
  padding: 30px 0;
}
</style>