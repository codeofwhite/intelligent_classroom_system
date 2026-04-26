<template>
  <div class="analysis-page">
    <h2>📊 课堂行为深度分析</h2>

    <!-- 信息卡片 -->
    <div class="card info-card">
      <div>班级：{{ report.class_name }}</div>
      <div>节次：{{ report.lesson_section }}</div>
      <div>总帧数：{{ stats.total_frames || 0 }}</div>
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

    <!-- ✅ AI 分析：按钮 + 加载 + MD渲染 + 展开/收起 -->
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



// 加载基础数据（不加载AI）
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

// ✅ 手动点击生成AI
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

// ✅ 简单MD渲染（支持标题、列表、换行）
function renderMarkdown(md) {
  if (!md) return ''
  let html = md
    .replace(/\n/g, '<br>')
    .replace(/### (.*?)(<br>|$)/g, '<h3>$1</h3>')
    .replace(/## (.*?)(<br>|$)/g, '<h2>$1</h2>')
    .replace(/# (.*?)(<br>|$)/g, '<h1>$1</h1>')
    .replace(/\d+\. (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')
    .replace(/- (.*?)(<br>|$)/g, '<div style="margin-left:16px">$1</div>')
  return html
}

// 初始化图表
function initCharts() {
  pieChart = echarts.init(pieRef.value)
  lineChart = echarts.init(lineRef.value)
  renderCharts()
}

// 绘制图表
function renderCharts() {
  if (!pieChart || !lineChart) return
  const data = stats.value.behavior_counts || {}
  const pieData = []
  for (const name in data) {
    pieData.push({ name, value: data[name] })
  }

  // 饼图
  pieChart.setOption({
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        data: pieData
      }
    ]
  })

  // 折线图（模拟专注度）
  lineChart.setOption({
    xAxis: { type: 'category', data: ['0s', '20s', '40s', '60s', '80s', '100s'] },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'line',
        smooth: true,
        data: [95, 88, 82, 76, 85, 90]
      }
    ]
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

.ai-markdown {
  line-height: 1.7;
  font-size: 15px;
  color: #333;
}
</style>