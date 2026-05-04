<template>
  <div class="schedule-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <h2>📅 我的课程安排</h2>
      <div class="header-meta">
        <el-tag effect="light" type="primary">本周 {{ schedule.length }} 节课</el-tag>
        <el-tag effect="light" type="success">已录制 {{ reports.length }} 份报告</el-tag>
      </div>
    </div>

    <!-- 周课程表 -->
    <div class="week-grid-card">
      <div class="week-header">
        <div class="time-col">时间</div>
        <div
          v-for="d in weekDays"
          :key="d.value"
          class="day-col"
          :class="{ 'is-today': d.value === todayWeekDay }"
        >
          <span>{{ d.label }}</span>
          <span v-if="d.value === todayWeekDay" class="today-dot">今天</span>
        </div>
      </div>

      <div class="week-body">
        <div v-for="section in sectionList" :key="section" class="time-row">
          <div class="time-col">
            <span class="section-num">第 {{ section }} 节</span>
            <span class="section-time">{{ sectionTimeMap[section] || '' }}</span>
          </div>
          <div
            v-for="d in weekDays"
            :key="d.value"
            class="day-col"
            :class="{ 'is-today': d.value === todayWeekDay }"
          >
            <div
              v-if="getClass(d.value, section)"
              class="course-card"
              :class="{ 'has-report': getMatchedReports(getClass(d.value, section)).length > 0 }"
              @click="handleCourseClick(getClass(d.value, section))"
            >
              <p class="course-name">{{ getClass(d.value, section).course_name }}</p>
              <p class="class-info">{{ getClass(d.value, section).class_name }}</p>
              <p class="classroom">{{ getClass(d.value, section).classroom }}</p>
              <span
                v-if="getMatchedReports(getClass(d.value, section)).length > 0"
                class="report-badge"
              >
                📊 {{ getMatchedReports(getClass(d.value, section)).length }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部两栏 -->
    <div class="bottom-section">
      <!-- 今日课程 -->
      <div class="today-card">
        <h3>📍 今日课程</h3>
        <div v-if="todayClasses.length === 0" class="empty-tip">今天没有课程安排</div>
        <div v-for="item in todayClasses" :key="item.section" class="today-item">
          <div class="today-time">第 {{ item.section }} 节</div>
          <div class="today-info">
            <p class="today-course">{{ item.course_name }}</p>
            <p class="today-detail">{{ item.class_name }} · {{ item.classroom }}</p>
          </div>
          <div class="today-actions">
            <el-button
              type="primary"
              size="small"
              @click="goToReportDetail(getMatchedReports(item)[0])"
              :disabled="getMatchedReports(item).length === 0"
            >
              {{ getMatchedReports(item).length > 0 ? `查看报告(${getMatchedReports(item).length})` : '暂无报告' }}
            </el-button>
            <el-button size="small" @click="$router.push('/videos')">
              录制
            </el-button>
          </div>
        </div>
      </div>

      <!-- 最近课堂报告 -->
      <div class="recent-card">
        <h3>📊 最近课堂报告</h3>
        <div v-if="reports.length === 0" class="empty-tip">暂无课堂报告</div>
        <div
          v-for="r in reports.slice(0, 8)"
          :key="r.id"
          class="recent-item"
          @click="goToReportDetail(r)"
        >
          <div class="recent-dot"></div>
          <div class="recent-info">
            <p class="recent-title">{{ r.class_name }} · {{ r.lesson_section }}</p>
            <p class="recent-time">{{ formatTime(r.created_at) }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 快捷操作 -->
    <div class="quick-actions">
      <el-button type="primary" @click="$router.push('/videos')">
        ▶️ 上传视频分析
      </el-button>
      <el-button @click="$router.push('/videos')">
        📊 查看所有课堂报告
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

const schedule = ref([])
const reports = ref([])

const weekDays = [
  { value: 1, label: '周一' },
  { value: 2, label: '周二' },
  { value: 3, label: '周三' },
  { value: 4, label: '周四' },
  { value: 5, label: '周五' },
  { value: 6, label: '周六' },
  { value: 7, label: '周日' },
]

const sectionTimeMap = {
  1: '08:00-08:45', 2: '08:55-09:40',
  3: '10:00-10:45', 4: '10:55-11:40',
  5: '14:00-14:45', 6: '14:55-15:40',
  7: '16:00-16:45', 8: '16:55-17:40',
}

const todayWeekDay = new Date().getDay() || 7

const sectionList = computed(() => {
  const sections = [...new Set(schedule.value.map(item => item.section))]
  return sections.sort((a, b) => a - b)
})

const todayClasses = computed(() => {
  return schedule.value.filter(item => item.week_day === todayWeekDay)
})

const getClass = (weekDay, section) => {
  return schedule.value.find(item => item.week_day === weekDay && item.section === section)
}

/**
 * 匹配逻辑：用 class_code + course_name 对应 lesson_section
 * 例如课表中 class_code=1, course_name="语文" → 匹配报告中 class_code=1, lesson_section="语文"
 */
const getMatchedReports = (courseItem) => {
  if (!courseItem) return []
  return reports.value.filter(r => {
    // class_code 必须匹配
    if (courseItem.class_code && r.class_code && String(courseItem.class_code) === String(r.class_code)) {
      // lesson_section 包含 course_name 即算匹配
      if (!courseItem.course_name || !r.lesson_section) return true
      return r.lesson_section.includes(courseItem.course_name) ||
             courseItem.course_name.includes(r.lesson_section)
    }
    // 兜底：class_name 匹配
    if (r.class_name === courseItem.class_name) {
      if (!courseItem.course_name || !r.lesson_section) return true
      return r.lesson_section.includes(courseItem.course_name) ||
             courseItem.course_name.includes(r.lesson_section)
    }
    return false
  })
}

const formatTime = (t) => {
  if (!t) return ''
  if (typeof t === 'string') return t.replace('T', ' ').slice(0, 16)
  return String(t).slice(0, 16)
}

const handleCourseClick = (course) => {
  const matched = getMatchedReports(course)
  if (matched.length > 0) {
    goToReportDetail(matched[0])
  }
}

const goToReportDetail = (report) => {
  if (!report) return
  router.push({ path: '/analysis-detail', query: { reportId: report.id } })
}

const loadData = async () => {
  const user = JSON.parse(localStorage.getItem('userInfo') || '{}')
  const teacher_code = user.teacher_code || 'T2025001'

  try {
    const { data } = await axios.post('http://localhost:5002/api/teacher/schedule_with_reports', {
      teacher_code
    })
    schedule.value = data.schedule || []
    reports.value = data.reports || []
  } catch (e) {
    console.error('加载数据失败', e)
    // 降级：只加载课表
    try {
      const res = await axios.post('http://localhost:5002/api/teacher/course_schedule', {
        teacher_code
      })
      schedule.value = res.data.list || []
    } catch (e2) {
      console.error('加载课表也失败', e2)
    }
  }
}

onMounted(() => loadData())
</script>

<style scoped>
.schedule-page {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 22px;
  color: #303133;
}

.header-meta {
  display: flex;
  gap: 10px;
}

/* 周课程表网格 */
.week-grid-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  margin-bottom: 24px;
  overflow-x: auto;
}

.week-header {
  display: grid;
  grid-template-columns: 100px repeat(7, 1fr);
  gap: 1px;
  margin-bottom: 1px;
}

.week-header .day-col,
.week-header .time-col {
  text-align: center;
  padding: 12px 8px;
  font-weight: 600;
  color: #606266;
  font-size: 14px;
  background: #f5f7fa;
  border-radius: 6px;
  margin: 0 1px;
}

.week-header .day-col.is-today {
  background: #ecf5ff;
  color: #409eff;
}

.today-dot {
  display: block;
  font-size: 11px;
  color: #409eff;
  margin-top: 2px;
}

.week-body .time-row {
  display: grid;
  grid-template-columns: 100px repeat(7, 1fr);
  gap: 1px;
  margin-bottom: 1px;
}

.time-row .time-col {
  text-align: center;
  padding: 16px 8px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.section-num {
  font-weight: 600;
  color: #303133;
  font-size: 13px;
}

.section-time {
  font-size: 11px;
  color: #c0c4cc;
  margin-top: 2px;
}

.time-row .day-col {
  min-height: 80px;
  padding: 4px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
  margin: 0 1px;
}

.time-row .day-col.is-today {
  background: #fafbff;
  border-color: #d9ecff;
}

.course-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border-radius: 8px;
  padding: 10px;
  height: 100%;
  cursor: default;
  position: relative;
  transition: transform 0.2s, box-shadow 0.2s;
}

.course-card.has-report {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  cursor: pointer;
}

.course-card.has-report:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.course-name {
  margin: 0 0 4px 0;
  font-size: 13px;
  font-weight: 600;
}

.class-info {
  margin: 0;
  font-size: 11px;
  opacity: 0.85;
}

.classroom {
  margin: 2px 0 0 0;
  font-size: 11px;
  opacity: 0.7;
}

.report-badge {
  position: absolute;
  top: 4px;
  right: 6px;
  font-size: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 10px;
  padding: 1px 6px;
}

/* 底部两栏 */
.bottom-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.today-card,
.recent-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.today-card h3,
.recent-card h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
  color: #303133;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.empty-tip {
  color: #c0c4cc;
  text-align: center;
  padding: 24px 0;
  font-size: 14px;
}

.today-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px dashed #f0f0f0;
}

.today-item:last-child {
  border-bottom: none;
}

.today-time {
  font-weight: 600;
  color: #409eff;
  font-size: 14px;
  min-width: 70px;
}

.today-info {
  flex: 1;
}

.today-course {
  margin: 0;
  font-weight: 600;
  font-size: 14px;
  color: #303133;
}

.today-detail {
  margin: 2px 0 0 0;
  font-size: 12px;
  color: #909399;
}

.today-actions {
  display: flex;
  gap: 6px;
}

.recent-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0;
  cursor: pointer;
  transition: background 0.2s;
  border-radius: 6px;
  padding-left: 8px;
}

.recent-item:hover {
  background: #f5f7fa;
}

.recent-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #409eff;
  flex-shrink: 0;
}

.recent-title {
  margin: 0;
  font-size: 14px;
  color: #303133;
}

.recent-time {
  margin: 2px 0 0 0;
  font-size: 12px;
  color: #c0c4cc;
}

.quick-actions {
  display: flex;
  gap: 12px;
}
</style>