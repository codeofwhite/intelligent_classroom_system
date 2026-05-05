<template>
  <div class="video-analysis-page">
    <div class="header">
      <h2>🎓 智能课堂行为分析系统</h2>
      <p>按班级、课程、时间管理课堂分析记录</p>
    </div>

    <!-- 实时视频预览 -->
    <div class="card" v-if="showLive">
      <div class="live-header">
        <h3>📹 实时课堂画面</h3>
        <button class="btn-sm" @click="showLive = false">隐藏预览</button>
      </div>
      <div class="live-container">
        <img src="http://localhost:5002/video_feed" class="live-video" @error="liveError = true" @load="liveError = false" />
        <div v-if="liveError" class="live-error">
          <p>📷 实时视频流未连接</p>
          <p>请确认模型推理服务已启动（端口 5002）</p>
        </div>
      </div>
      <p class="live-tip">💡 此为终端设备实时画面，可在课堂终端页面开始录制</p>
    </div>

    <div v-if="!showLive" class="card">
      <button class="btn-sm" @click="showLive = true" style="margin-bottom:12px">📹 显示实时画面</button>
    </div>

    <!-- 上传分析 -->
    <div class="card">
      <h3>📂 上传课堂视频进行分析</h3>
      <div class="upload-form">

        <select v-model="classId" class="select-input">
          <option value="">-- 选择班级 --</option>
          <option v-for="c in classList" :key="c.class_code" :value="c.class_code">
            {{ c.class_name }}
          </option>
        </select>

        <input v-model="lessonSection" class="select-input" placeholder="节次：如 第1节" />
        <input v-model="recordDate" type="date" class="select-input" />

        <input type="file" ref="videoInput" @change="onFileChange" accept="video/*" />

        <select v-model="selectedModel" class="select-input" @change="switchModel" :disabled="switching">
          <option v-for="model in modelOptions" :key="model" :value="model">
            {{ model }}
          </option>
        </select>

        <button @click="startAnalysis" :disabled="loading || switching" class="btn-primary">
          {{ loading ? "AI 分析中..." : "开始上传并分析" }}
        </button>
      </div>

      <div v-if="resultUrl || statistics" class="result-section">
        <h4>✅ 分析完成</h4>
        <video v-if="resultUrl" :src="resultUrl" controls class="result-video"></video>

        <div v-if="statistics" class="stats-section">
          <h4>📊 课堂行为统计</h4>
          <table class="behavior-table">
            <tr>
              <th>行为</th>
              <th>次数</th>
            </tr>
            <tr v-for="(count, name) in statistics.behavior_counts" :key="name">
              <td>{{ name }}</td>
              <td>{{ count }}</td>
            </tr>
          </table>
        </div>
      </div>
    </div>

    <!-- 历史记录 -->
    <div class="card">
      <h3>📜 课堂分析历史记录</h3>
      <button @click="fetchTeacherReports" class="btn-sm" style="margin-left:10px;">
        🔄 刷新历史记录
      </button>

      <div v-for="group in dateClassGroups" :key="group.key" class="collapse-card">
        <div class="collapse-header" @click="toggleGroup(group.key)">
          <span>📅 {{ group.date }} | {{ group.className }}</span>
          <span>{{ group.list.length }} 条记录</span>
        </div>

        <div v-show="group.open" class="collapse-body">
          <div v-for="item in group.list" :key="item.id" class="lesson-item">
            <div>
              <div class="label">课程：{{ item.lesson_section }}</div>
              <div class="time">{{ item.created_at }}</div>
            </div>

            <div class="action-buttons">
              <button class="btn-sm" @click="goToAnalysis(item)">查看分析报告</button>
              <!-- 🔥 删除按钮 -->
              <button class="btn-sm btn-delete" @click.stop="deleteReport(item)">
                删除
              </button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="Object.keys(dateClassGroups).length === 0" class="empty">
        暂无分析记录
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const videoInput = ref(null)
const videoFile = ref(null)
const loading = ref(false)
const switching = ref(false)
const resultUrl = ref('')
const statistics = ref(null)

const modelOptions = ref([])
const selectedModel = ref('')

const classId = ref('')
const lessonSection = ref('第1节')
const recordDate = ref('')
const teacherCode = ref('T2025001')

const reportList = ref([])
const openGroups = ref({})

const classList = ref([])

// 实时预览
const showLive = ref(false)
const liveError = ref(false)

// 加载所有班级
const loadClassList = async () => {
  try {
    const { data } = await axios.get('http://localhost:5002/api/class/list')
    classList.value = data.list || []
  } catch (err) {
    console.error('加载班级失败', err)
  }
}

const dateClassGroups = computed(() => {
  const groups = {}
  reportList.value.forEach(r => {
    const date = r.created_at.split(' ')[0]
    const realClassCode = r.class_code || r.class_id
    const key = `${date}_${realClassCode}`

    // 安全查找班级
    const cls = classList.value.find(item => item.class_code === realClassCode)
    const className = cls ? cls.class_name : '未知班级'

    if (!groups[key]) {
      groups[key] = {
        key: key,
        date: date,
        classId: realClassCode,
        className: className,
        list: [],
        open: openGroups.value[key] || false
      }
    }
    groups[key].list.push(r)
  })
  return groups
})

function toggleGroup(key) {
  openGroups.value[key] = !openGroups.value[key]
}

// 刷新报告
async function fetchTeacherReports() {
  try {
    const res = await axios.get('http://localhost:5002/api/teacher/reports', {
      params: {
        teacher_code: teacherCode.value,
        _t: Date.now()
      },
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    })
    reportList.value = res.data
  } catch (err) {
    console.error("加载报告失败", err)
    reportList.value = []
  }
}

// 上传分析
function onFileChange(e) {
  videoFile.value = e.target.files[0]
}

async function startAnalysis() {
  console.log("当前选中班级：", classId.value)
  if (!videoFile.value) return alert('请选择视频')
  if (!classId.value) return alert('请选择班级')
  if (!recordDate.value) return alert('请选择日期')

  loading.value = true
  const fd = new FormData()
  fd.append('video', videoFile.value)
  fd.append('teacher_code', teacherCode.value)
  fd.append('class_code', classId.value)
  fd.append('lesson_section', lessonSection.value)
  fd.append('lesson_date', recordDate.value) // ✅ 加这行！

  try {
    const res = await axios.post('http://localhost:5002/upload_video', fd)
    statistics.value = res.data.statistics
    alert('上传分析完成')
    fetchTeacherReports()
  } finally {
    loading.value = false
  }
}

// 进入报告
function goToAnalysis(item) {
  router.push({
    path: '/analysis-detail',
    query: { reportId: item.id }
  })
}

// 🔥 删除报告（完整功能）
async function deleteReport(item) {
  if (!confirm(`确定要删除【${item.lesson_section}】的分析报告吗？\n删除后无法恢复！`)) {
    return
  }

  try {
    await axios.post('http://localhost:5002/api/report/delete', {
      report_id: item.id
    })
    alert('删除成功！')
    fetchTeacherReports()
  } catch (err) {
    alert('删除失败')
    console.error(err)
  }
}

// 模型切换
async function fetchModels() {
  const r = await axios.get('http://localhost:5002/get_models')
  modelOptions.value = r.data.models
  selectedModel.value = r.data.current
}

async function switchModel() {
  switching.value = true
  await axios.post('http://localhost:5002/switch_model', {
    model_name: selectedModel.value
  })
  switching.value = false
}

onMounted(() => {
  fetchModels()
  fetchTeacherReports()
  loadClassList() // ✅ 加这行
})
</script>

<style scoped>
.video-analysis-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px 20px;
  background: #f7f9fc;
  min-height: 100vh;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.upload-form {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}

.select-input {
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

.result-video {
  width: 100%;
  border-radius: 10px;
  margin-top: 10px;
}

.behavior-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

.behavior-table th,
td {
  border: 1px solid #eee;
  padding: 10px;
  text-align: center;
}

.collapse-card {
  margin-bottom: 12px;
  border: 1px solid #eee;
  border-radius: 12px;
  overflow: hidden;
}

.collapse-header {
  padding: 14px 18px;
  background: #f9fafb;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  font-weight: 500;
}

.collapse-body {
  padding: 14px 18px;
}

.lesson-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.btn-sm {
  padding: 6px 12px;
  background: #e6f7ff;
  border: 1px solid #91d5ff;
  border-radius: 6px;
  cursor: pointer;
}

/* 🔥 删除按钮样式 */
.btn-delete {
  background: #fff2f2;
  border-color: #ffadad;
  color: #e53935;
}

.time {
  font-size: 12px;
  color: #999;
}

.empty {
  text-align: center;
  padding: 30px;
  color: #999;
}

/* 实时预览 */
.live-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.live-header h3 {
  margin: 0;
}

.live-container {
  width: 100%;
  max-width: 640px;
  border-radius: 8px;
  overflow: hidden;
  border: 2px solid #e5e6eb;
  background: #000;
  position: relative;
}

.live-video {
  width: 100%;
  display: block;
}

.live-error {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #111;
  color: #999;
}

.live-error p {
  margin: 4px 0;
}

.live-tip {
  margin: 8px 0 0 0;
  font-size: 12px;
  color: #999;
}
</style>