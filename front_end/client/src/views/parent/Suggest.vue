<template>
  <div class="advice-page">
    <div class="page-header">
      <h2>🤝 家校共育 · AI 成长分析</h2>
      <p>基于历史行为数据 · 智能生成个性化教育方案</p>
    </div>

    <!-- 选择孩子 -->
    <div class="select-box" v-if="studentList.length > 0">
      <label>选择孩子：</label>
      <select v-model="currentCode" @change="onChildChange">
        <option v-for="s in studentList" :key="s.student_code" :value="s.student_code">
          {{ s.student_name }}
        </option>
      </select>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-card">
      <div class="loading-spinner"></div>
      <p>正在分析孩子课堂数据...</p>
    </div>

    <!-- 数据概览 -->
    <div class="overview-card" v-if="!loading && currentCode">
      <div class="ov-item">
        <div class="ov-icon">📊</div>
        <div class="ov-val">{{ totalReports }}</div>
        <div class="ov-label">课堂报告</div>
      </div>
      <div class="ov-item">
        <div class="ov-icon">🎯</div>
        <div class="ov-val">{{ avgFocus }}%</div>
        <div class="ov-label">平均专注度</div>
      </div>
      <div class="ov-item">
        <div class="ov-icon">📈</div>
        <div class="ov-val">{{ levelText }}</div>
        <div class="ov-label">综合评价</div>
      </div>
    </div>

    <!-- 优势与待改进 -->
    <div class="two-col" v-if="!loading && currentCode">
      <div class="mini-card green-card">
        <h4>✅ 表现优势</h4>
        <ul>
          <li v-for="s in strengths" :key="s">{{ s }}</li>
        </ul>
      </div>
      <div class="mini-card orange-card">
        <h4>⚠️ 待改进项</h4>
        <ul>
          <li v-for="s in weaknesses" :key="s">{{ s }}</li>
        </ul>
      </div>
    </div>

    <!-- AI 行为分析 -->
    <div class="card analysis-card" v-if="!loading && currentCode">
      <div class="card-icon">📈</div>
      <h3>课堂行为综合分析</h3>
      <div class="card-text" v-if="summary">{{ summary }}</div>
      <div class="card-text placeholder" v-else>暂无分析数据</div>
    </div>

    <!-- AI 共育建议 -->
    <div class="card advice-card" v-if="!loading && currentCode">
      <div class="card-icon">🤖</div>
      <h3>AI 个性化共育建议</h3>
      <div class="advice-sections" v-if="advice">
        <div class="advice-block school">
          <h4>🏫 在校建议</h4>
          <p>{{ schoolAdvice }}</p>
        </div>
        <div class="advice-block home">
          <h4>🏠 家庭建议</h4>
          <p>{{ homeAdvice }}</p>
        </div>
      </div>
      <div class="card-text placeholder" v-else>暂无建议</div>
    </div>

    <!-- 家校协同说明 -->
    <div class="card tip-card">
      <div class="card-icon">📩</div>
      <h3>家校协同说明</h3>
      <div class="tip-list">
        <div class="tip-item">
          <span class="tip-num">1</span>
          <p>教师会定期同步课堂状态，家长可根据 AI 建议进行家庭引导</p>
        </div>
        <div class="tip-item">
          <span class="tip-num">2</span>
          <p>实现学校与家庭双向共育，帮助孩子持续提升课堂表现与学习习惯</p>
        </div>
        <div class="tip-item">
          <span class="tip-num">3</span>
          <p>建议每周查看一次报告，关注孩子专注度变化趋势</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'

let parentUser = null
try {
  parentUser = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

const studentList = ref([])
const currentCode = ref('')
const summary = ref('')
const advice = ref('')
const loading = ref(false)
const reportList = ref([])

// 统计数据
const totalReports = computed(() => reportList.value.length)
const avgFocus = computed(() => {
  if (reportList.value.length === 0) return 0
  return Math.round(reportList.value.reduce((t, r) => t + (r.focus_rate || 0), 0) / reportList.value.length)
})
const levelText = computed(() => {
  const v = avgFocus.value
  if (v >= 90) return '优秀'
  if (v >= 80) return '良好'
  if (v >= 70) return '一般'
  return '需努力'
})

// 优势项
const strengths = computed(() => {
  if (reportList.value.length === 0) return []
  const avgHand = Math.round(reportList.value.reduce((t, r) => t + (r.raised_hand || 0), 0) / reportList.value.length)
  const avgDown = Math.round(reportList.value.reduce((t, r) => t + (r.looking_down || 0), 0) / reportList.value.length)
  const result = []
  if (avgFocus.value >= 80) result.push('专注度表现良好')
  if (avgHand >= 3) result.push('课堂互动积极')
  if (avgDown <= 3) result.push('学习态度端正')
  if (result.length === 0) result.push('正在积累课堂数据')
  return result
})

// 待改进
const weaknesses = computed(() => {
  if (reportList.value.length === 0) return []
  const avgDown = Math.round(reportList.value.reduce((t, r) => t + (r.looking_down || 0), 0) / reportList.value.length)
  const result = []
  if (avgFocus.value < 70) result.push('课堂专注度需提升')
  if (avgDown > 5) result.push('低头分心次数偏多')
  if (avgFocus.value >= 70 && avgFocus.value < 80) result.push('专注度有提升空间')
  if (result.length === 0) result.push('继续保持当前状态')
  return result
})

// 拆分 AI 建议为在校/家庭两部分
const schoolAdvice = computed(() => {
  if (!advice.value) return ''
  const parts = advice.value.split(/家庭|家长/)
  return parts[0] || '建议老师多关注孩子课堂表现，适时给予鼓励。'
})
const homeAdvice = computed(() => {
  if (!advice.value) return ''
  const parts = advice.value.split(/家庭|家长/)
  if (parts.length > 1) return '家长' + parts[1]
  return '建议家长配合老师，关注孩子课后学习状态，营造良好的学习环境。'
})

onMounted(async () => {
  if (!parentUser || parentUser.role !== 'parent') return
  try {
    const res = await axios.post('http://localhost:5001/parent-children', {
      user_code: parentUser.user_code
    })
    studentList.value = res.data.children || []
    if (studentList.value.length > 0) {
      currentCode.value = studentList.value[0].student_code
      loadAll()
    }
  } catch (err) {
    console.error(err)
  }
})

function onChildChange() {
  loadAll()
}

async function loadAll() {
  if (!currentCode.value) return
  loading.value = true
  try {
    // 并行加载报告和 AI 建议
    const [reportRes, adviceRes] = await Promise.all([
      axios.get('http://localhost:5002/api/student/my-reports', {
        params: { student_code: currentCode.value }
      }),
      axios.get('http://localhost:5002/api/ai/advice', {
        params: { student_code: currentCode.value }
      }).catch(() => ({ data: { summary: '暂无数据', advice: '暂无建议' } }))
    ])
    reportList.value = reportRes.data.list || []
    summary.value = adviceRes.data.summary || ''
    advice.value = adviceRes.data.advice || ''
  } catch (err) {
    console.error(err)
    reportList.value = []
    summary.value = '暂无数据'
    advice.value = '暂无建议'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.advice-page {
  padding: 20px;
  background: linear-gradient(to bottom, #f9faff, #f1f5ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

.page-header { text-align: center; margin-bottom: 20px; }
.page-header h2 { font-size: 22px; font-weight: 600; margin: 0 0 6px 0; color: #2c3e50; }
.page-header p { font-size: 13px; color: #7f8c8d; margin: 0; }

.select-box { text-align: center; margin-bottom: 18px; }
.select-box label { margin-right: 8px; font-size: 14px; color: #555; }
.select-box select {
  padding: 8px 14px; border-radius: 12px; border: 1px solid #e2e8ff;
  background: #fff; font-size: 14px;
}

/* 加载 */
.loading-card {
  background: #fff; border-radius: 16px; padding: 40px;
  text-align: center; margin-bottom: 16px;
}
.loading-spinner {
  width: 36px; height: 36px; border: 3px solid #e0e0e0;
  border-top-color: #7c5fff; border-radius: 50%;
  animation: spin 0.8s linear infinite; margin: 0 auto 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-card p { color: #999; font-size: 14px; margin: 0; }

/* 数据概览 */
.overview-card {
  background: linear-gradient(135deg, #7c5fff, #9b77ff);
  border-radius: 16px; padding: 18px; margin-bottom: 16px;
  display: flex; justify-content: space-around; text-align: center; color: white;
}
.ov-item { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.ov-icon { font-size: 20px; }
.ov-val { font-size: 20px; font-weight: 700; }
.ov-label { font-size: 11px; opacity: 0.85; }

/* 双栏 */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.mini-card {
  background: #fff; border-radius: 14px; padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.mini-card h4 { margin: 0 0 8px 0; font-size: 14px; }
.mini-card ul { margin: 0; padding: 0 0 0 16px; }
.mini-card li { font-size: 13px; color: #555; margin-bottom: 4px; }
.green-card { border-left: 3px solid #52c41a; }
.green-card h4 { color: #52c41a; }
.orange-card { border-left: 3px solid #fa8c16; }
.orange-card h4 { color: #fa8c16; }

/* 卡片 */
.card {
  background: #fff; border-radius: 16px; padding: 18px;
  margin-bottom: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.card-icon { font-size: 22px; margin-bottom: 6px; }
.card h3 { margin: 0 0 12px 0; font-size: 16px; font-weight: 600; color: #2c3e50; }
.card-text {
  font-size: 14px; line-height: 1.7; color: #444;
  background: #f8f9fd; padding: 14px; border-radius: 12px;
}
.card-text.placeholder { color: #ccc; text-align: center; }

.analysis-card { border-left: 4px solid #7c5fff; }
.advice-card { border-left: 4px solid #50c878; }

/* 建议分区 */
.advice-sections { display: flex; flex-direction: column; gap: 12px; }
.advice-block {
  padding: 14px; border-radius: 12px;
}
.advice-block h4 { margin: 0 0 8px 0; font-size: 14px; }
.advice-block p { margin: 0; font-size: 13px; line-height: 1.6; color: #444; }
.school { background: #f0f7ff; }
.school h4 { color: #1677ff; }
.home { background: #f6ffed; }
.home h4 { color: #52c41a; }

/* 协同说明 */
.tip-card { border-left: 4px solid #ffb946; background: #fffbf5; }
.tip-list { display: flex; flex-direction: column; gap: 10px; }
.tip-item { display: flex; gap: 10px; align-items: flex-start; }
.tip-num {
  width: 22px; height: 22px; background: #ffb946; color: #fff;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.tip-item p { margin: 0; font-size: 13px; color: #555; line-height: 1.5; }
</style>