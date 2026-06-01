<template>
  <div class="report-page">
    <!-- 头部 -->
    <div class="header">
      <h2>👨‍👩‍👧‍👦 孩子课堂行为报告</h2>
      <p>实时行为统计 · 学习状态分析 · 成长可视化</p>
    </div>

    <!-- 选择孩子 -->
    <div class="student-selector">
      <button 
        v-for="s in studentList" 
        :key="s.student_code" 
        class="student-btn"
        :class="{ active: selectedCode === s.student_code }"
        @click="loadStudentReport(s.student_code)"
      >
        👦 {{ s.student_name }}
      </button>
    </div>

    <!-- 数据概览（选中孩子后显示） -->
    <div class="overview-card" v-if="selectedCode && reportList.length > 0">
      <div class="overview-item">
        <span>总报告</span>
        <strong>{{ reportList.length }}</strong>
      </div>
      <div class="overview-item">
        <span>平均专注</span>
        <strong>{{ avgFocus }}%</strong>
      </div>
      <div class="overview-item">
        <span>评价</span>
        <strong>{{ levelText }}</strong>
      </div>
    </div>

    <!-- 筛选栏 -->
    <div class="filter-bar" v-if="selectedCode && reportList.length > 0">
      <div class="filter-tabs">
        <button v-for="f in filters" :key="f.key" class="filter-tab"
          :class="{ active: currentFilter === f.key }" @click="currentFilter = f.key">
          {{ f.label }}
        </button>
      </div>
      <span class="filter-count">{{ filteredReports.length }} 条</span>
    </div>

    <!-- 按月分组的报告列表 -->
    <div v-for="group in monthGroups" :key="group.key" class="month-group" v-if="selectedCode">
      <div class="month-header" @click="group.open = !group.open">
        <span class="month-title">📅 {{ group.label }}</span>
        <span class="month-meta">{{ group.items.length }} 节 · 均 {{ group.avgFocus }}%</span>
        <span class="expand-arrow">{{ group.open ? '▼' : '▶' }}</span>
      </div>
      <div v-show="group.open" class="month-body">
        <div class="report-item" v-for="item in group.items" :key="item.id" @click="openDetail(item)">
          <div class="item-left">
            <div class="item-time">{{ formatTime(item.lesson_time) }}</div>
            <div class="item-stats">
              坐姿 {{ item.normal_posture }} · 举手 {{ item.raised_hand }} · 低头 {{ item.looking_down }}
            </div>
          </div>
          <div class="item-right">
            <div class="focus-badge" :class="focusBadgeClass(item.focus_rate)">
              {{ item.focus_rate }}%
            </div>
            <span class="arrow">›</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div class="empty-tip" v-if="!selectedCode">
      请选择孩子查看课堂报告
    </div>
    <div class="empty-tip" v-else-if="reportList.length === 0">
      暂无课堂报告数据
    </div>

    <!-- 详情弹窗 -->
    <div class="modal" v-if="currentDetail" @click="closeDetail">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>📊 课堂行为详情</h3>
          <button @click="closeDetail">×</button>
        </div>

        <div class="detail-body">
          <div class="info-row">
            <span>课程时间</span>
            <span>{{ currentDetail.lesson_time }}</span>
          </div>
          <div class="info-row">
            <span>专注度</span>
            <span class="purple">{{ currentDetail.focus_rate }}%</span>
          </div>

          <!-- 行为统计（自适应显示） -->
          <div class="stats" v-if="currentDetail.behaviors_json">
            <div v-for="(cnt, label) in parseBehaviors(currentDetail.behaviors_json)" :key="label">
              <label>{{ label }}</label>
              <span>{{ cnt }}</span>
            </div>
          </div>
          <div class="stats" v-else>
            <div>
              <label>正常坐姿</label>
              <span>{{ currentDetail.normal_posture }}</span>
            </div>
            <div>
              <label>举手次数</label>
              <span>{{ currentDetail.raised_hand }}</span>
            </div>
            <div>
              <label>低头次数</label>
              <span class="red">{{ currentDetail.looking_down }}</span>
            </div>
          </div>

          <!-- AI 建议 -->
          <div class="suggest" v-if="currentDetail.ai_comment">
            <h4>💡 AI 学习建议</h4>
            <p>{{ currentDetail.ai_comment }}</p>
          </div>

          <!-- 老师评语 -->
          <div class="comment" v-if="currentDetail.teacher_comment">
            <h4>👨‍🏫 老师评语</h4>
            <p>{{ currentDetail.teacher_comment }}</p>
          </div>
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
const selectedCode = ref(null)
const reportList = ref([])
const currentDetail = ref(null)
const currentFilter = ref('all')

const filters = [
  { key: 'all', label: '全部' },
  { key: 'high', label: '优秀 ≥85' },
  { key: 'medium', label: '良好 70-84' },
  { key: 'low', label: '需关注 <70' }
]

const avgFocus = computed(() => {
  if (reportList.value.length === 0) return 0
  return Math.round(reportList.value.reduce((t, i) => t + (i.focus_rate || 0), 0) / reportList.value.length)
})
const levelText = computed(() => {
  if (avgFocus.value >= 90) return '优秀'
  if (avgFocus.value >= 80) return '良好'
  if (avgFocus.value >= 70) return '一般'
  return '需努力'
})

const filteredReports = computed(() => {
  if (currentFilter.value === 'all') return reportList.value
  return reportList.value.filter(r => {
    const f = r.focus_rate || 0
    if (currentFilter.value === 'high') return f >= 85
    if (currentFilter.value === 'medium') return f >= 70 && f < 85
    return f < 70
  })
})

const monthGroups = computed(() => {
  const groups = {}
  filteredReports.value.forEach(r => {
    const d = new Date(r.lesson_time)
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    if (!groups[key]) {
      groups[key] = { key, label: `${d.getFullYear()}年${d.getMonth() + 1}月`, items: [], open: true, avgFocus: 0 }
    }
    groups[key].items.push(r)
  })
  Object.values(groups).forEach(g => {
    g.avgFocus = Math.round(g.items.reduce((t, i) => t + (i.focus_rate || 0), 0) / g.items.length)
  })
  return Object.values(groups).sort((a, b) => b.key.localeCompare(a.key))
})

const focusBadgeClass = (rate) => rate >= 85 ? 'good' : rate >= 70 ? 'medium' : 'bad'
const formatTime = (t) => {
  if (!t) return ''
  const d = new Date(t)
  return `${d.getMonth()+1}月${d.getDate()}日 周${'日一二三四五六'[d.getDay()]} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`
}
const parseBehaviors = (s) => { try { return JSON.parse(s) } catch { return {} } }

onMounted(async () => {
  if (!parentUser || parentUser.role !== 'parent') { alert('请以家长身份登录'); return }
  try {
    const res = await axios.post('http://localhost:5001/parent-children', { user_code: parentUser.user_code })
    studentList.value = res.data.children
  } catch (err) { console.error(err) }
})

async function loadStudentReport(student_code) {
  selectedCode.value = student_code
  currentFilter.value = 'all'
  try {
    const res = await axios.get('http://localhost:5002/api/student/my-reports', { params: { student_code } })
    reportList.value = res.data.list || []
  } catch (e) { console.error(e) }
}

const openDetail = (item) => { currentDetail.value = item }
const closeDetail = () => { currentDetail.value = null }
</script>

<style scoped>
.report-page {
  padding: 20px;
  background: linear-gradient(to bottom, #f9faff, #f1f5ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.header { text-align: center; margin-bottom: 20px; }
.header h2 { font-size: 22px; font-weight: 600; margin: 0 0 6px 0; color: #2c3e50; }
.header p { font-size: 13px; color: #7f8c8d; margin: 0; }

.student-selector { display: flex; gap: 10px; justify-content: center; margin-bottom: 18px; flex-wrap: wrap; }
.student-btn {
  padding: 10px 18px; border: 1px solid #e2e8ff; background: #fff;
  border-radius: 12px; cursor: pointer; font-size: 14px; transition: all 0.2s;
}
.student-btn.active {
  background: linear-gradient(90deg, #7c5fff, #9b77ff); color: white;
  border-color: #7c5fff; transform: scale(1.03);
}

/* 概览卡片 */
.overview-card {
  background: linear-gradient(135deg, #7c5fff, #9b77ff);
  border-radius: 16px; padding: 16px; margin-bottom: 16px;
  display: flex; justify-content: space-around; text-align: center; color: white;
}
.overview-item { display: flex; flex-direction: column; gap: 4px; }
.overview-item span { font-size: 12px; opacity: 0.85; }
.overview-item strong { font-size: 18px; }

/* 筛选 */
.filter-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.filter-tabs { display: flex; gap: 6px; }
.filter-tab {
  padding: 6px 12px; border: 1px solid #e0e0e0; border-radius: 16px;
  background: #fff; font-size: 12px; cursor: pointer; color: #666; transition: all 0.2s;
}
.filter-tab.active { background: #7c5fff; color: #fff; border-color: #7c5fff; }
.filter-count { font-size: 12px; color: #999; }

/* 月度分组 */
.month-group { margin-bottom: 12px; border-radius: 14px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.month-header {
  background: #fff; padding: 14px 16px; display: flex; align-items: center;
  gap: 10px; cursor: pointer; border-bottom: 1px solid #f0f0f0;
}
.month-title { font-weight: 600; font-size: 14px; color: #2c3e50; }
.month-meta { flex: 1; font-size: 12px; color: #999; text-align: right; }
.expand-arrow { font-size: 11px; color: #bbb; }
.month-body { background: #fafbff; }

.report-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; border-bottom: 1px solid #f0f2f5; cursor: pointer; transition: background 0.2s;
}
.report-item:hover { background: #f0f5ff; }
.report-item:last-child { border-bottom: none; }
.item-time { font-size: 14px; font-weight: 500; color: #2c3e50; margin-bottom: 3px; }
.item-stats { font-size: 12px; color: #999; }
.item-right { display: flex; align-items: center; gap: 8px; }
.focus-badge { padding: 4px 10px; border-radius: 12px; font-size: 14px; font-weight: 600; }
.focus-badge.good { background: #e6ffed; color: #00b42a; }
.focus-badge.medium { background: #fff7e6; color: #fa8c16; }
.focus-badge.bad { background: #fff2f0; color: #ff4d4f; }
.arrow { color: #ccc; font-size: 18px; }

.empty-tip { text-align: center; color: #999; padding: 60px 20px; font-size: 15px; }

/* 弹窗 */
.modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center; z-index: 999; backdrop-filter: blur(4px);
}
.modal-content {
  background: #fff; width: 90%; max-width: 420px; border-radius: 20px;
  max-height: 85vh; overflow-y: auto; box-shadow: 0 16px 32px rgba(0,0,0,0.2); animation: modalUp 0.3s ease;
}
@keyframes modalUp { from { transform: translateY(30px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.modal-header {
  padding: 18px 20px; border-bottom: 1px solid #f1f1f1;
  display: flex; justify-content: space-between; align-items: center;
}
.modal-header h3 { margin: 0; font-size: 17px; font-weight: 600; color: #2c3e50; }
.modal-header button {
  background: #f5f6fa; border: none; width: 30px; height: 30px; border-radius: 50%;
  font-size: 16px; cursor: pointer; color: #7f8c8d; display: flex; align-items: center; justify-content: center;
}
.detail-body { padding: 20px; }
.info-row { display: flex; justify-content: space-between; padding: 10px 0; font-size: 14px; color: #2c3e50; }
.purple { color: #7c5fff !important; font-weight: 600; }
.red { color: #ff4757 !important; }
.stats {
  background: #f8f9fd; border-radius: 12px; padding: 14px;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 16px 0;
}
.stats div { text-align: center; }
.stats label { font-size: 11px; color: #999; display: block; margin-bottom: 4px; }
.stats span { font-weight: 600; font-size: 15px; color: #2c3e50; }
.suggest, .comment { margin-bottom: 16px; }
.suggest h4, .comment h4 { font-size: 14px; font-weight: 600; margin: 0 0 8px 0; color: #2c3e50; }
.suggest p, .comment p {
  background: #f8f9fd; padding: 14px; border-radius: 12px;
  font-size: 13px; line-height: 1.6; margin: 0; color: #34495e;
}
</style>