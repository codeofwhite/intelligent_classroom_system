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
        <input type="file" ref="videoInput" @change="onFileChange" accept="video/*" />

        <select v-model="selectedModel" class="model-select" @change="switchModel" :disabled="switching">
          <option v-for="model in modelOptions" :key="model" :value="model">
            {{ model }}
          </option>
        </select>

        <button @click="startAnalysis" :disabled="loading || switching" class="btn-primary">
          {{ loading ? "AI 分析中..." : "开始上传并分析" }}
        </button>
      </div>

      <!-- 结果区域：只要有 视频 或 统计 就显示 -->
      <div v-if="resultUrl || statistics" class="result-section">
        <h4>✅ 分析结果</h4>

        <!-- 视频：有 URL 才显示 -->
        <video
          v-if="resultUrl"
          :src="resultUrl"
          controls
          class="result-video"
          muted
        ></video>

        <!-- AI 行为统计 -->
        <div v-if="statistics" class="stats-section">
          <h4>📊 课堂行为统计（6 类）</h4>

          <table class="behavior-table">
            <thead>
              <tr>
                <th>行为类别</th>
                <th>出现总次数</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(count, name) in statistics.behavior_counts" :key="name">
                <td>{{ name }}</td>
                <td>{{ count }}</td>
              </tr>
            </tbody>
          </table>
        </div>
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
          <video :src="video.url" controls muted></video>
          <p class="name">{{ video.name }}</p>

          <!-- ✅ 查看历史结果（同时加载视频 + 统计） -->
          <button class="small-btn" @click="loadHistoryStats(video)">
            查看分析结果
          </button>
        </div>
        <div v-if="videoList.length === 0" class="empty">暂无历史记录</div>
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
const switching = ref(false)
const resultUrl = ref('')
const videoList = ref([])

const statistics = ref(null)
const jsonUrl = ref('')

const modelOptions = ref([])
const selectedModel = ref('')

const studentList = ref([
  { id: 101, name: '张小明', headUp: 96, focus: 92, status: '正常' },
  { id: 102, name: '李华', headUp: 88, focus: 85, status: '偶尔走神' },
  { id: 103, name: '王磊', headUp: 72, focus: 68, status: '注意力低' },
  { id: 104, name: '刘芳', headUp: 98, focus: 95, status: '优秀' },
])

function onFileChange(e) {
  videoFile.value = e.target.files[0]
}

async function fetchModels() {
  try {
    const res = await axios.get('http://localhost:5002/get_models')
    modelOptions.value = res.data.models
    selectedModel.value = res.data.current
  } catch (err) {
    console.error('获取模型失败')
  }
}

async function switchModel() {
  if (!selectedModel.value) return
  switching.value = true
  try {
    await axios.post('http://localhost:5002/switch_model', {
      model_name: selectedModel.value
    })
  } catch (err) {
    alert('切换失败')
  } finally {
    switching.value = false
  }
}

async function startAnalysis() {
  if (!videoFile.value) {
    alert('请选择视频')
    return
  }
  loading.value = true
  const formData = new FormData()
  formData.append('video', videoFile.value)

  try {
    const res = await axios.post('http://localhost:5002/upload_video', formData)
    resultUrl.value = res.data.video_url
    statistics.value = res.data.statistics
  } catch (err) {
    alert('分析失败')
  } finally {
    loading.value = false
  }
}

async function fetchVideoList() {
  try {
    const res = await axios.get('http://localhost:5002/list_videos')
    videoList.value = res.data
  } catch (err) {
    console.error('获取失败')
  }
}


async function loadHistoryStats(video) {
  try {
    const videoName = video.name
    const videoUrl = video.url

    // 构造统计文件名
    const statName = videoName.replace('video_', 'stats_').replace('.mp4', '.json')
    const res = await axios.get(`http://localhost:5002/get_history_stat/${statName}`)

    // ✅ 同时赋值：视频 + 统计
    resultUrl.value = videoUrl
    statistics.value = res.data

    alert('已加载历史分析结果！')
  } catch (e) {
    alert('暂无该视频的统计数据')
    console.error(e)
  }
}

onMounted(() => {
  fetchModels()
  fetchVideoList()
})
</script>

<style scoped>
.video-analysis-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px 20px;
  background: #f7f9fc;
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}
.header h2 {
  font-size: 24px;
  color: #222;
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

.upload-area {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

.model-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.btn-primary {
  background: #42b983;
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
}
.btn-primary:disabled {
  background: #a0d9bc;
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
}

.stats-section {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #eee;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.stat-card {
  background: #f9fafb;
  padding: 14px;
  border-radius: 10px;
  text-align: center;
}
.stat-card .label {
  font-size: 13px;
  color: #666;
}
.stat-card .num {
  font-size: 22px;
  font-weight: bold;
  margin-top: 4px;
}
.behavior-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 15px;
}
.behavior-table th, .behavior-table td {
  border: 1px solid #eee;
  padding: 10px;
  text-align: center;
}
.behavior-table th {
  background: #f5f7fa;
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
}

.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}
.video-card {
  border: 1px solid #eee;
  border-radius: 12px;
  padding: 12px;
}
.video-card video {
  width: 100%;
  border-radius: 8px;
}
.small-btn {
  margin-top: 8px;
  width: 100%;
  background: #e6f7ff;
  border: 1px solid #91d5ff;
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
.empty {
  text-align: center;
  color: #999;
  padding: 30px 0;
  grid-column: 1/-1;
}
</style>