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

        <!-- 模型选择 -->
        <select v-model="selectedModel" class="model-select" @change="switchModel">
          <option value="headup">抬头率识别模型</option>
          <option value="focus">专注度分析模型</option>
          <option value="behavior">综合行为分析模型</option>
        </select>

        <button @click="startAnalysis" :disabled="loading" class="btn-primary">
          {{ loading ? "AI 分析中..." : "开始上传并分析" }}
        </button>
      </div>

      <!-- 结果回放 -->
      <div v-if="resultUrl" class="result-section">
        <h4>✅ 分析完成</h4>
        <video :src="resultUrl" controls class="result-video"></video>
        <p class="tip">文件已自动保存至云端存储</p>

        <!-- 学生行为列表 -->
        <div class="student-behavior-wrapper" v-if="studentList.length > 0">
          <h4>👥 本节课学生行为概览</h4>
          <table class="student-table">
            <thead>
              <tr>
                <th>姓名</th>
                <th>抬头率</th>
                <th>专注度</th>
                <th>行为状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="stu in studentList" :key="stu.id">
                <td>{{ stu.name }}</td>
                <td>{{ stu.headUp }}%</td>
                <td>{{ stu.focus }}%</td>
                <td>{{ stu.status }}</td>
                <td>
                  <button class="report-btn" @click="goToReport(stu.id)">
                    查看行为报告
                  </button>
                </td>
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
          <video :src="video.url" controls></video>
          <p class="name">{{ video.name }}</p>
          <button class="small-btn" @click="goToReportByVideo(video.id)">
            查看全班报告
          </button>
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
import { useRouter } from 'vue-router'

const router = useRouter()
const videoInput = ref(null)
const videoFile = ref(null)
const loading = ref(false)
const resultUrl = ref('')
const videoList = ref([])

// 模型选择（对接 /switch_model）
const selectedModel = ref('behavior')

// 模拟学生行为数据
const studentList = ref([
  { id: 101, name: '张小明', headUp: 96, focus: 92, status: '正常' },
  { id: 102, name: '李华', headUp: 88, focus: 85, status: '偶尔走神' },
  { id: 103, name: '王磊', headUp: 72, focus: 68, status: '注意力低' },
  { id: 104, name: '刘芳', headUp: 98, focus: 95, status: '优秀' },
])

// 选择文件
function onFileChange(e) {
  videoFile.value = e.target.files[0]
}

// ======================
// ✅ 切换模型（调用你的接口）
// ======================
async function switchModel() {
  try {
    await axios.post('http://localhost:5000/switch_model', {
      model_name: selectedModel.value
    })
    console.log('模型切换成功：', selectedModel.value)
  } catch (err) {
    alert('模型切换失败')
  }
}

// ======================
// ✅ 上传视频 + 分析
// ======================
async function startAnalysis() {
  if (!videoFile.value) {
    alert('请选择视频')
    return
  }

  loading.value = true
  const formData = new FormData()
  formData.append('video', videoFile.value)
  
  // 把当前选中模型一起带给后端
  formData.append('model', selectedModel.value)

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

// 跳转到报告页
function goToReport(studentId) {
  router.push({
    path: '/student-report',
    query: { studentId }
  })
}

function goToReportByVideo(videoId) {
  router.push({ path: '/student-report', query: { videoId } })
}

onMounted(() => {
  fetchVideoList()
})
</script>

<style scoped>
.video-analysis-page {
  max-width: 1200px;
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

.model-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  outline: none;
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

.student-behavior-wrapper {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #eee;
}
.student-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
.student-table th,
.student-table td {
  padding: 10px;
  text-align: center;
  border: 1px solid #eee;
}
.student-table th {
  background: #f5f7fa;
}
.report-btn {
  background: #1890ff;
  color: white;
  border: none;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
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
.small-btn {
  margin-top: 8px;
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.empty {
  color: #999;
  grid-column: 1 / -1;
  text-align: center;
  padding: 30px 0;
}
</style>