<template>
  <div class="report-page">
    <!-- 头部 -->
    <div class="header">
      <h2>📋 我的课堂行为报告</h2>
      <p>查看每节课的表现与成长建议</p>
    </div>

    <!-- 本周概况 -->
    <div class="week-card">
      <h3>本周概况</h3>
      <div class="items">
        <div>
          <span>上课节数</span>
          <strong>{{ weekReportCount }} 节</strong>
        </div>
        <div>
          <span>平均专注度</span>
          <strong>{{ weekAvgFocus }}%</strong>
        </div>
        <div>
          <span>总体评价</span>
          <strong>{{ weekLevel }}</strong>
        </div>
      </div>
    </div>

    <!-- 筛选栏 -->
    <div class="filter-bar">
      <div class="filter-tabs">
        <button
          v-for="f in filters"
          :key="f.key"
          class="filter-tab"
          :class="{ active: currentFilter === f.key }"
          @click="currentFilter = f.key"
        >
          {{ f.label }}
        </button>
      </div>
      <span class="filter-count">共 {{ filteredReports.length }} 条</span>
    </div>

    <!-- 按月分组的报告列表 -->
    <div v-for="group in monthGroups" :key="group.key" class="month-group">
      <div class="month-header" @click="group.open = !group.open">
        <span class="month-title">📅 {{ group.label }}</span>
        <span class="month-meta">{{ group.items.length }} 节课 · 均 {{ group.avgFocus }}%</span>
        <span class="expand-arrow">{{ group.open ? '▼' : '▶' }}</span>
      </div>
      <div v-show="group.open" class="month-body">
        <div
          class="report-item"
          v-for="item in group.items"
          :key="item.id"
          @click="openDetail(item)"
        >
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

    <div v-if="filteredReports.length === 0" class="empty">
      暂无课堂报告数据
    </div>

    <!-- 弹窗：单节课详情 -->
    <div class="modal" v-if="currentDetail" @click="closeDetail">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>课堂行为详情</h3>
          <button @click="closeDetail">×</button>
        </div>

        <div class="detail-body">
          <div class="info-row">
            <span>课程时间</span>
            <span>{{ currentDetail.lesson_time }}</span>
          </div>
          <div class="info-row">
            <span>专注度</span>
            <span class="blue">{{ currentDetail.focus_rate }}%</span>
          </div>

          <!-- 行为统计（自适应显示） -->
          <div class="stats" v-if="currentDetail.behaviors_json">
            <div v-for="(cnt, label) in parseBehaviors(currentDetail.behaviors_json)" :key="label">
              <label>{{ label }}</label>
              <span>{{ cnt }}</span>
            </div>
          </div>
          <!-- 兼容旧数据 -->
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
  <StudentAiFloat></StudentAiFloat>
</template>

<script setup>
import { ref, computed, onMounted, reactive } from 'vue'
import axios from 'axios'
import StudentAiFloat from '../../components/StudentAiFloat.vue'

const reportList = ref([])
const currentDetail = ref(null)
const currentFilter = ref('all')

const filters = [
  { key: 'all', label: '全部' },
  { key: 'high', label: '优秀 ≥85' },
  { key: 'medium', label: '良好 70-84' },
  { key: 'low', label: '需关注 <70' }
]

let user = null
try {
  user = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}
const student_code = user?.student_code

// 本周数据
const weekReportCount = computed(() => {
  return weekReports.value.length
})
const weekAvgFocus = computed(() => {
  if (weekReports.value.length === 0) return 0
  const sum = weekReports.value.reduce((t, i) => t + (i.focus_rate || 0), 0)
  return Math.round(sum / weekReports.value.length)
})
const weekLevel = computed(() => {
  const avg = weekAvgFocus.value
  if (avg >= 90) return '优秀'
  if (avg >= 80) return '良好'
  if (avg >= 70) return '一般'
  return '需努力'
})

const weekReports = computed(() => {
  const now = new Date()
  const startOfWeek = new Date(now)
  startOfWeek.setDate(now.getDate() - now.getDay() + 1)
  startOfWeek.setHours(0, 0, 0, 0)
  return reportList.value.filter(r => new Date(r.lesson_time) >= startOfWeek)
})

// 筛选
const filteredReports = computed(() => {
  if (currentFilter.value === 'all') return reportList.value
  return reportList.value.filter(r => {
    const f = r.focus_rate || 0
    if (currentFilter.value === 'high') return f >= 85
    if (currentFilter.value === 'medium') return f >= 70 && f < 85
    if (currentFilter.value === 'low') return f < 70
    return true
  })
})

// 按月分组
const monthGroups = computed(() => {
  const groups = {}
  filteredReports.value.forEach(r => {
    const d = new Date(r.lesson_time)
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    if (!groups[key]) {
      groups[key] = {
        key,
        label: `${d.getFullYear()}年${d.getMonth() + 1}月`,
        items: [],
        open: true,
        avgFocus: 0
      }
    }
    groups[key].items.push(r)
  })
  // 计算每月平均
  Object.values(groups).forEach(g => {
    const sum = g.items.reduce((t, i) => t + (i.focus_rate || 0), 0)
    g.avgFocus = Math.round(sum / g.items.length)
  })
  return Object.values(groups).sort((a, b) => b.key.localeCompare(a.key))
})

const focusBadgeClass = (rate) => {
  if (rate >= 85) return 'good'
  if (rate >= 70) return 'medium'
  return 'bad'
}

const formatTime = (t) => {
  if (!t) return ''
  const d = new Date(t)
  const month = d.getMonth() + 1
  const day = d.getDate()
  const hours = String(d.getHours()).padStart(2, '0')
  const mins = String(d.getMinutes()).padStart(2, '0')
  const week = '日一二三四五六'[d.getDay()]
  return `${month}月${day}日 周${week} ${hours}:${mins}`
}

const parseBehaviors = (jsonStr) => {
  try {
    return JSON.parse(jsonStr)
  } catch {
    return {}
  }
}

const loadMyReports = async () => {
  if (!student_code) {
    alert('未获取到学生信息')
    return
  }
  try {
    const res = await axios.get('http://localhost:5002/api/student/my-reports', {
      params: { student_code }
    })
    reportList.value = res.data.list || []
  } catch (err) {
    console.error('加载报告失败', err)
  }
}

const openDetail = (item) => { currentDetail.value = item }
const closeDetail = () => { currentDetail.value = null }

onMounted(() => { loadMyReports() })
</script>

<style scoped>
.report-page {
  padding: 20px;
  background: linear-gradient(to bottom, #f8faff, #eff4ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

.header { text-align: center; margin-bottom: 20px; }
.header h2 { font-size: 22px; font-weight: 600; margin: 0 0 4px 0; color: #2c3e50; }
.header p { font-size: 13px; color: #7f8c8d; margin: 0; }

/* 本周概况 */
.week-card {
  background: linear-gradient(135deg, #4285f4, #64a6ff);
  border-radius: 16px;
  padding: 18px;
  margin-bottom: 16px;
  color: white;
  box-shadow: 0 6px 16px rgba(66, 133, 244, 0.25);
}
.week-card h3 { font-size: 15px; margin: 0 0 12px 0; font-weight: 500; }
.items { display: flex; justify-content: space-around; text-align: center; }
.items div { display: flex; flex-direction: column; gap: 4px; }
.items span { font-size: 12px; opacity: 0.85; }
.items strong { font-size: 18px; font-weight: 600; }

/* 筛选栏 */
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.filter-tabs { display: flex; gap: 6px; }
.filter-tab {
  padding: 6px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 16px;
  background: #fff;
  font-size: 12px;
  cursor: pointer;
  color: #666;
  transition: all 0.2s;
}
.filter-tab.active {
  background: #4285f4;
  color: #fff;
  border-color: #4285f4;
}
.filter-count { font-size: 12px; color: #999; }

/* 月度分组 */
.month-group {
  margin-bottom: 12px;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.month-header {
  background: #fff;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
}
.month-title { font-weight: 600; font-size: 14px; color: #2c3e50; }
.month-meta { flex: 1; font-size: 12px; color: #999; text-align: right; }
.expand-arrow { font-size: 11px; color: #bbb; }

.month-body { background: #fafbff; }

/* 报告项 */
.report-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f2f5;
  cursor: pointer;
  transition: background 0.2s;
}
.report-item:hover { background: #f0f5ff; }
.report-item:last-child { border-bottom: none; }

.item-time { font-size: 14px; font-weight: 500; color: #2c3e50; margin-bottom: 3px; }
.item-stats { font-size: 12px; color: #999; }

.item-right { display: flex; align-items: center; gap: 8px; }

.focus-badge {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 14px;
  font-weight: 600;
}
.focus-badge.good { background: #e6ffed; color: #00b42a; }
.focus-badge.medium { background: #fff7e6; color: #fa8c16; }
.focus-badge.bad { background: #fff2f0; color: #ff4d4f; }

.arrow { color: #ccc; font-size: 18px; }

.empty { text-align: center; padding: 40px; color: #999; font-size: 14px; }

/* 弹窗 */
.modal {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center;
  z-index: 999; backdrop-filter: blur(4px);
}
.modal-content {
  background: #fff; width: 90%; max-width: 420px;
  border-radius: 20px; max-height: 85vh; overflow-y: auto;
  box-shadow: 0 16px 32px rgba(0,0,0,0.2);
  animation: modalUp 0.3s ease;
}
@keyframes modalUp {
  from { transform: translateY(30px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
.modal-header {
  padding: 18px 20px; border-bottom: 1px solid #f1f1f1;
  display: flex; justify-content: space-between; align-items: center;
}
.modal-header h3 { margin: 0; font-size: 17px; font-weight: 600; color: #2c3e50; }
.modal-header button {
  background: #f5f6fa; border: none; width: 30px; height: 30px;
  border-radius: 50%; font-size: 16px; cursor: pointer; color: #7f8c8d;
  display: flex; align-items: center; justify-content: center;
}

.detail-body { padding: 20px; }
.info-row {
  display: flex; justify-content: space-between;
  padding: 10px 0; font-size: 14px; color: #2c3e50;
}
.blue { color: #4285f4 !important; font-weight: 600; }
.red { color: #ff4757 !important; }

.stats {
  background: #f8f9fd; border-radius: 12px; padding: 14px;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  margin: 16px 0;
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