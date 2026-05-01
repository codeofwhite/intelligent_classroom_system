<template>
  <div class="analysis-page">
    <h2>📊 课堂行为深度分析</h2>

    <!-- 信息卡片 + 绑定按钮 -->
    <div class="card info-card">
      <div>班级：{{ report.class_name }}</div>
      <div>节次：{{ report.lesson_section }}</div>
      <div>总帧数：{{ stats.total_frames || 0 }}</div>
      <button @click="openBindPopup"
        style="padding:8px 16px; background:#ff9800; color:white; border:none; border-radius:8px;cursor:pointer">
        👤 学生身份绑定
      </button>
    </div>

    <!-- 视频 -->
    <div class="card">
      <h3>🎥 分析视频</h3>
      <video v-if="videoUrl" :src="videoUrl" controls class="video"></video>
    </div>

    <!-- 双图表布局 -->
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

    <!-- AI 分析 -->
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

    <!-- 行为表格 -->
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

    <!-- 🔥 学生身份绑定弹窗 -->
    <div v-if="showBindPopup"
      style="position:fixed; left:0; top:0; width:100vw; height:100vh; background:rgba(0,0,0,0.5); display:flex; justify-content:center; align-items:center; z-index:9999;">
      <div style="background:white; padding:24px; border-radius:12px; width:420px;">
        <h3 style="margin-top:0;">👤 绑定学生身份</h3>
        <div v-for="sid in studentIds" :key="sid" style="margin:12px 0;">
          <label style="font-weight:bold;">追踪 ID：{{ sid }}</label>
          <input v-model="bindForm[sid]" placeholder="请输入学生姓名"
            style="width:100%; padding:10px; margin-top:6px; border:1px solid #ddd; border-radius:8px;">
        </div>
        <div style="text-align:right; margin-top:20px;">
          <button @click="saveStudentBind"
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

// 🔥 绑定弹窗
const showBindPopup = ref(false)
const studentIds = ref([])
const bindForm = ref({})

// 加载详情
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

// 生成AI
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

// 打开绑定弹窗
async function openBindPopup() {
  showBindPopup.value = true
  const res = await axios.get('http://localhost:5002/api/report/students', {
    params: { id: reportId }
  })
  studentIds.value = res.data.student_ids || []
  bindForm.value = {}
}

// 保存绑定
async function saveStudentBind() {
  for (const sid of studentIds.value) {
    const name = bindForm.value[sid] || `学生${sid}`
    await axios.post('http://localhost:5002/api/report/bind_student', {
      report_id: reportId,
      track_id: sid,
      student_name: name
    })
  }
  alert('✅ 学生身份绑定成功！')
  showBindPopup.value = false
}

// MD渲染（完整版：支持加粗、标题、列表、换行）
function renderMarkdown(md) {
  if (!md) return ''

  let html = md
    // 1. 加粗 **文字** → <strong>
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // 2. 换行
    .replace(/\n/g, '<br>')
    // 3. 标题
    .replace(/### (.*?)(<br>|$)/g, '<h3>$1</h3>')
    .replace(/## (.*?)(<br>|$)/g, '<h2>$1</h2>')
    .replace(/# (.*?)(<br>|$)/g, '<h1>$1</h1>')
    // 4. 有序/无序列表
    .replace(/\d+\. (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')
    .replace(/- (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')

  return html
}

// 图表
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