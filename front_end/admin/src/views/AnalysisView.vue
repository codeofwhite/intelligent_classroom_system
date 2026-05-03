<template>
  <div class="analysis-page">
    <h2>📊 课堂行为深度分析</h2>

    <div class="card info-card">
      <div>班级：{{ report.class_name }}</div>
      <div>节次：{{ report.lesson_section }}</div>
      <div>总帧数：{{ stats.total_frames || 0 }}</div>
      <button @click="openBindPopup"
        style="padding:8px 16px; background:#ff9800; color:white; border:none; border-radius:8px;cursor:pointer">
        👤 人脸身份绑定
      </button>
    </div>

    <div class="card">
      <h3>🎥 分析视频</h3>
      <video v-if="videoUrl" :src="videoUrl" controls class="video"></video>
    </div>

    <div class="chart-row">
      <div class="card chart-box">
        <h3>🥧 行为分布饼图</h3>
        <div ref="pieRef" style="width:100%;height:280px"></div>
      </div>

      <div class="card chart-box">
        <h3>📈 专注度趋势</h3>
        <div ref="lineRef" style="width:100%;height:280px"></div>
      </div>
    </div>

    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3>🤖 AI 智能课堂分析</h3>
        <button @click="generateAI" :disabled="loading"
          style="padding:8px 16px; background:#409eff; color:white; border:none; border-radius:8px;cursor:pointer">
          {{ loading ? "生成中..." : "点击生成AI分析" }}
        </button>
      </div>

      <div v-if="aiAnalysis" style="margin-top:16px;">
        <div :style="{ maxHeight: showFull ? 'none' : '180px', overflow: 'hidden' }" class="ai-markdown"
          v-html="renderMarkdown(aiAnalysis)">
        </div>
        <div style="text-align:right; margin-top:8px;">
          <button @click="showFull = !showFull" style="color:#409eff; background:none; border:none;cursor:pointer">
            {{ showFull ? "收起" : "展开更多" }}
          </button>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>📋 详细行为统计</h3>
      <table class="table">
        <tr>
          <th>行为</th>
          <th>次数</th>
        </tr>
        <tr v-for="(count, name) in stats.behavior_counts" :key="name">
          <td>{{ name }}</td>
          <td>{{ count }}</td>
        </tr>
      </table>
    </div>

    <!-- 🔥 人脸绑定学生（下拉选择学生ID版） -->
    <div v-if="showBindPopup"
      style="position:fixed; left:0; top:0; width:100vw; height:100vh; background:rgba(0,0,0,0.5); display:flex; justify-content:center; align-items:center; z-index:9999;">
      <div style="background:white; padding:24px; border-radius:12px; width:460px;">
        <h3 style="margin-top:0;">👤 人脸绑定学生</h3>

        <div v-for="fid in faceIds" :key="fid" style="margin:14px 0;">
          <label style="font-weight:bold;">人脸 ID：{{ fid }}</label>
          <select v-model="bindMap[fid]"
            style="width:100%; padding:10px; margin-top:6px; border:1px solid #ddd; border-radius:8px;">
            <option value="">-- 选择学生 --</option>
            <option :value="stu" v-for="stu in studentList" :key="stu.student_code">
              {{ stu.name }}
            </option>
          </select>
        </div>

        <div style="text-align:right; margin-top:20px;">
          <button @click="saveBind"
            style="padding:8px 16px; background:#409eff; color:white; border:none; border-radius:8px; margin-right:8px;">
            保存绑定
          </button>
          <button @click="showBindPopup = false" style="padding:8px 16px; border:1px solid #ddd; border-radius:8px;">
            取消
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import * as echarts from 'echarts'

const route = useRoute()
const reportId = route.query.reportId

const report = ref({})
const stats = ref({})
const videoUrl = ref('')

const aiAnalysis = ref('')
const loading = ref(false)
const showFull = ref(false)

const pieRef = ref(null)
const lineRef = ref(null)
let pieChart = null
let lineChart = null

const showBindPopup = ref(false)
const faceIds = ref([])
const studentList = ref([]) // 本班学生列表
const bindMap = ref({})     // 人脸ID → 学生对象

async function loadDetail() {
  const res = await axios.get('http://localhost:5002/api/report/detail', {
    params: { id: reportId }
  })
  report.value = res.data.report
  stats.value = res.data.statistics || {}
  aiAnalysis.value = res.data.ai_analysis || ''

  const u = await axios.get('http://localhost:5002/get_video_url', {
    params: { path: report.value.minio_video_path }
  })
  videoUrl.value = u.data
}

async function generateAI() {
  loading.value = true
  try {
    const res = await axios.post('http://localhost:5002/api/generate_and_save_ai', {
      id: reportId
    })
    aiAnalysis.value = res.data.ai_analysis
  } finally {
    loading.value = false
  }
}

// 🔥 打开人脸绑定弹窗
async function openBindPopup() {
  showBindPopup.value = true
  bindMap.value = {} // 先清空

  // 1. 获取本节课人脸ID
  const res = await axios.get('http://localhost:5002/api/report/face_ids', {
    params: { id: reportId }
  })
  faceIds.value = res.data.face_ids || []

  // 2. 获取本班学生列表
  const stuRes = await axios.get('http://localhost:5002/api/class/students', {
    params: { class_code: report.value.class_code }
  })
  studentList.value = stuRes.data.students || []

  // 3. 关键：加载本班已绑定的人脸→学生映射
  const mapRes = await axios.get('http://localhost:5002/api/face/mapping', {
    params: { class_code: report.value.class_code }
  })
  const existingMap = mapRes.data.map || {}

  // 4. 回填已绑定的学生
  for (const fid of faceIds.value) {
    if (existingMap[fid]) {
      // 找到对应的学生对象，塞给 bindMap
      const stu = studentList.value.find(s => s.student_code === existingMap[fid].student_code)
      if (stu) bindMap.value[fid] = stu
    }
  }
}

// 保存绑定（人脸 ↔ 学生ID）
async function saveBind() {
  for (const fid of faceIds.value) {
    const stu = bindMap.value[fid]
    if (!stu) continue

    await axios.post('http://localhost:5002/api/face/bind', {
      face_id: fid,
      student_code: stu.student_code,    // 绑定学生ID
      student_name: stu.name,
      class_code: report.value.class_code
    })
  }
  alert('✅ 人脸绑定学生成功！')
  showBindPopup.value = false
}

function renderMarkdown(md) {
  if (!md) return ''
  let html = md
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
    .replace(/### (.*?)(<br>|$)/g, '<h3>$1</h3>')
    .replace(/## (.*?)(<br>|$)/g, '<h2>$1</h2>')
    .replace(/# (.*?)(<br>|$)/g, '<h1>$1</h1>')
    .replace(/\d+\. (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')
    .replace(/- (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')
  return html
}

function initCharts() {
  pieChart = echarts.init(pieRef.value)
  lineChart = echarts.init(lineRef.value)
  renderCharts()
}

function renderCharts() {
  if (!pieChart || !lineChart) return
  const data = stats.value.behavior_counts || {}
  const pieData = []
  for (const name in data) {
    pieData.push({ name, value: data[name] })
  }

  pieChart.setOption({
    tooltip: { trigger: 'item' },
    series: [{ type: 'pie', radius: ['40%', '70%'], data: pieData }]
  })

  lineChart.setOption({
    xAxis: { type: 'category', data: ['0s', '20s', '40s', '60s', '80s', '100s'] },
    yAxis: { type: 'value' },
    series: [{ type: 'line', smooth: true, data: [95, 88, 82, 76, 85, 90] }]
  })
}

onMounted(async () => {
  await loadDetail()
  await nextTick()
  initCharts()
})

watch(stats, () => {
  renderCharts()
}, { deep: true })
</script>

<style scoped>
.analysis-page {
  max-width: 1400px;
  margin: auto;
  padding: 30px 20px;
  background: #f7f9fc;
  min-height: 100vh;
}

.card {
  background: white;
  padding: 22px;
  border-radius: 16px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

.info-card {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  font-size: 15px;
  align-items: center;
}

.video {
  width: 100%;
  border-radius: 12px;
}

.chart-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.chart-box {
  min-height: 320px;
}

.table {
  width: 100%;
  border-collapse: collapse;
}

.table th,
.table td {
  border: 1px solid #eee;
  padding: 12px;
  text-align: center;
}

.ai-markdown strong {
  font-weight: bold;
  color: #2c3e50;
}
</style>