<template>
  <div class="teacher-home">
    <!-- 顶部欢迎 -->
    <div class="home-header">
      <h2>👋 您好，{{ teacherName }}！欢迎使用智慧课堂分析系统</h2>
      <p class="date">{{ today }}</p>
    </div>

    <!-- 数据统计卡片 -->
    <div class="data-grid">
      <div class="stat-card purple">
        <p class="label">班级总人数</p>
        <h3>{{ totalStudents }}</h3>
        <span>本班学生总数</span>
      </div>

      <div class="stat-card blue">
        <p class="label">当前班级</p>
        <h3>{{ currentClass || '--' }}</h3>
        <span>{{ subjectName || '待分配' }}</span>
      </div>

      <div class="stat-card green">
        <p class="label">学生列表</p>
        <h3>{{ studentList.length }}</h3>
        <span>已录入学生</span>
      </div>
    </div>

    <!-- 快捷功能入口 -->
    <div class="quick-bar">
      <h3>🚀 快捷操作</h3>
      <div class="quick-grid">
        <button class="quick-btn primary" @click="goToClass">
          <span>▶️</span>
          开始上课
        </button>
        <button class="quick-btn" @click="goToMember">
          <span>👥</span>
          班级成员
        </button>
        <button class="quick-btn" @click="goToSchedule">
          <span>📅</span>
          课程安排
        </button>
        <button class="quick-btn" @click="goToReport">
          <span>📊</span>
          报告管理
        </button>
      </div>
    </div>

    <!-- 内容区域 -->
    <div class="action-sections">
      <!-- 实时课堂状态 -->
      <section class="action-block">
        <div class="block-header">
          <h3>📹 当前课堂信息</h3>
        </div>
        <div class="real-time-info">
          <div class="real-item">
            <label>当前班级</label>
            <span>{{ currentClass }}</span>
          </div>
          <div class="real-item">
            <label>授课科目</label>
            <span>{{ subjectName }}</span>
          </div>
          <div class="real-item">
            <label>班级人数</label>
            <span>{{ totalStudents }} 人</span>
          </div>
        </div>
        <div class="tip">
          计算机视觉分析已就绪，可实时监测学生抬头率、专注度、行为状态
        </div>
      </section>

      <!-- 学生列表（可折叠 + 搜索） -->
      <section class="action-block">
        <div class="block-header clickable" @click="studentListExpanded = !studentListExpanded">
          <h3>👨‍🎓 本班学生列表</h3>
          <span class="expand-icon">{{ studentListExpanded ? '▼' : '▶' }}</span>
        </div>

        <!-- 搜索栏 -->
        <div class="search-bar" v-show="studentListExpanded">
          <input
            v-model="studentSearch"
            type="text"
            placeholder="🔍 搜索学生姓名..."
            class="search-input"
          />
          <span class="search-count">{{ filteredStudentList.length }} / {{ studentList.length }}</span>
        </div>

        <!-- 学生表格 -->
        <div class="student-table-wrapper" v-show="studentListExpanded">
          <table class="report-table" v-if="filteredStudentList.length > 0">
            <thead>
              <tr>
                <th>序号</th>
                <th>学生姓名</th>
                <th>性别</th>
                <th>年龄</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, index) in pagedStudentList" :key="item.student_code">
                <td>{{ (studentPage - 1) * pageSize + index + 1 }}</td>
                <td>{{ item.name }}</td>
                <td>{{ item.gender }}</td>
                <td>{{ item.age || '--' }}</td>
                <td>
                  <span class="status done">正常</span>
                </td>
              </tr>
            </tbody>
          </table>
          <div v-else class="empty-tip">无匹配学生</div>

          <!-- 分页 -->
          <div class="pagination" v-if="filteredStudentList.length > pageSize">
            <button :disabled="studentPage <= 1" @click="studentPage--">‹ 上一页</button>
            <span>{{ studentPage }} / {{ totalStudentPages }}</span>
            <button :disabled="studentPage >= totalStudentPages" @click="studentPage++">下一页 ›</button>
          </div>
        </div>

        <!-- 折叠时的摘要 -->
        <div class="collapsed-summary" v-show="!studentListExpanded">
          <p>共 {{ totalStudents }} 名学生，点击展开查看列表</p>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

// 老师信息
const teacherName = ref('')
const currentClass = ref('')
const subjectName = ref('')
const today = ref('')

// 统计数据
const totalStudents = ref(0)

// 学生列表
const studentList = ref([])

// 学生列表交互状态
const studentListExpanded = ref(false)
const studentSearch = ref('')
const studentPage = ref(1)
const pageSize = 10

// 搜索过滤
const filteredStudentList = computed(() => {
  const keyword = studentSearch.value.trim().toLowerCase()
  if (!keyword) return studentList.value
  return studentList.value.filter(s =>
    s.name && s.name.toLowerCase().includes(keyword)
  )
})

// 分页
const totalStudentPages = computed(() => Math.max(1, Math.ceil(filteredStudentList.value.length / pageSize)))
const pagedStudentList = computed(() => {
  const start = (studentPage.value - 1) * pageSize
  return filteredStudentList.value.slice(start, start + pageSize)
})

// 搜索时重置页码
watch(studentSearch, () => { studentPage.value = 1 })

// 格式化日期
const formatDate = () => {
  const d = new Date()
  const year = d.getFullYear()
  const month = d.getMonth() + 1
  const day = d.getDate()
  const week = '星期' + '日一二三四五六'[d.getDay()]
  return `${year}年${month}月${day}日 ${week}`
}

// 加载老师信息 + 班级 + 学生
const loadTeacherData = async () => {
  const userInfo = JSON.parse(localStorage.getItem('userInfo'))
  if (!userInfo) return

  teacherName.value = userInfo.name
  const userCode = userInfo.user_code

  try {
    // 1. 获取老师的班级、科目
    const { data } = await axios.post('http://localhost:5001/teacher-class', {
      user_code: userCode
    })

    currentClass.value = data.class_name
    subjectName.value = data.subject

    // 2. 获取班级学生总数
    totalStudents.value = data.student_count

    // 3. 获取学生列表
    studentList.value = data.students

  } catch (err) {
    console.error('加载失败', err)
  }
}

onMounted(() => {
  today.value = formatDate()
  loadTeacherData()
})

const router = useRouter()

const goToClass = () => router.push('/videos')
const goToSchedule = () => router.push('/schedule')
const goToMember = () => router.push({ name: 'members' })
const goToReport = () => router.push('/reports')
</script>

<style scoped>
.teacher-home {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.home-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.home-header h2 {
  margin: 0;
  font-size: 22px;
  color: #333;
}

.date {
  color: #666;
  font-size: 14px;
}

.data-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  background: #fff;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.stat-card.blue { border-left: 4px solid #1890ff; }
.stat-card.green { border-left: 4px solid #52c41a; }
.stat-card.purple { border-left: 4px solid #722ed1; }

.stat-card .label {
  color: #666;
  margin: 0 0 8px 0;
  font-size: 14px;
}

.stat-card h3 {
  font-size: 30px;
  margin: 0 0 4px 0;
  color: #333;
}

.stat-card span {
  font-size: 12px;
  color: #999;
}

.quick-bar {
  background: white;
  padding: 16px 20px;
  border-radius: 10px;
  margin-bottom: 24px;
}

.quick-bar h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.quick-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.quick-btn {
  padding: 16px;
  border: none;
  border-radius: 8px;
  background: #f7f8fa;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.quick-btn.primary {
  background: #1890ff;
  color: white;
}

.quick-btn span {
  font-size: 20px;
}

.action-sections {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.action-block {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.block-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.block-header.clickable {
  cursor: pointer;
  user-select: none;
}

.block-header.clickable:hover {
  background: #fafafa;
}

.block-header h3 {
  margin: 0;
  font-size: 16px;
}

.expand-icon {
  font-size: 12px;
  color: #999;
  transition: transform 0.2s;
}

/* 搜索栏 */
.search-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.search-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #1890ff;
}

.search-count {
  font-size: 12px;
  color: #999;
  white-space: nowrap;
}

/* 表格容器 */
.student-table-wrapper {
  max-height: 400px;
  overflow-y: auto;
}

.report-table {
  width: 100%;
  border-collapse: collapse;
}

.report-table th,
.report-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
  font-size: 13px;
}

.report-table th {
  position: sticky;
  top: 0;
  background: #fafafa;
  z-index: 1;
}

.status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.status.done {
  background: #f6ffed;
  color: #52c41a;
}

/* 分页 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
  padding-top: 12px;
}

.pagination button {
  padding: 4px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
}

.pagination button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.pagination span {
  font-size: 13px;
  color: #666;
}

/* 折叠摘要 */
.collapsed-summary {
  padding: 16px;
  text-align: center;
  color: #999;
  font-size: 14px;
  background: #fafafa;
  border-radius: 8px;
  cursor: pointer;
}

.collapsed-summary:hover {
  background: #f0f5ff;
  color: #1890ff;
}

.empty-tip {
  text-align: center;
  padding: 24px;
  color: #c0c4cc;
  font-size: 14px;
}

.real-time-info {
  margin-bottom: 16px;
}

.real-item {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px dashed #eee;
}

.real-item label {
  color: #666;
  font-weight: 500;
}

.tip {
  background: #e6f7ff;
  color: #1890ff;
  padding: 10px;
  border-radius: 6px;
  font-size: 13px;
  margin-top: 10px;
}
</style>