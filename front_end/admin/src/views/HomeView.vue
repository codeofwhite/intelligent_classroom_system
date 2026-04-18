<template>
  <div class="teacher-home">
    <!-- 顶部欢迎 -->
    <div class="home-header">
      <h2>👋 您好，{{ teacherName }}！欢迎使用智慧课堂分析系统</h2>
      <p class="date">{{ today }}</p>
    </div>

    <!-- 数据统计卡片 -->
    <div class="data-grid">
      <div class="stat-card blue">
        <p class="label">今日课程</p>
        <h3>{{ todayCourse }}</h3>
        <span>节</span>
      </div>

      <div class="stat-card green">
        <p class="label">实时平均抬头率</p>
        <h3>{{ realTimeLookUp }}%</h3>
        <span>CV实时分析</span>
      </div>

      <div class="stat-card orange">
        <p class="label">课堂专注度</p>
        <h3>{{ focusRate }}%</h3>
        <span>综合评分</span>
      </div>

      <div class="stat-card red">
        <p class="label">异常行为学生</p>
        <h3>{{ abnormalStudents }}</h3>
        <span>走神/低头/讲话</span>
      </div>

      <div class="stat-card purple">
        <p class="label">班级总人数</p>
        <h3>{{ totalStudents }}</h3>
        <span>本班学生总数</span>
      </div>

      <div class="stat-card cyan">
        <p class="label">待推送报告</p>
        <h3>{{ waitPushReports }}</h3>
        <span>家长端可接收</span>
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
        <button class="quick-btn" @click="goToClassGroup">
          <span>🏫</span>
          班级信息
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

      <!-- 学生列表 -->
      <section class="action-block">
        <div class="block-header">
          <h3>👨‍🎓 本班学生列表</h3>
        </div>
        <table class="report-table">
          <thead>
            <tr>
              <th>学生姓名</th>
              <th>班级</th>
              <th>性别</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in studentList" :key="item.id">
              <td>{{ item.name }}</td>
              <td>{{ currentClass }}</td>
              <td>{{ item.gender }}</td>
              <td>
                <span class="status done">正常</span>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// 老师信息
const teacherName = ref('')
const currentClass = ref('')
const subjectName = ref('')
const today = ref('')

// 统计数据
const todayCourse = ref(4)
const realTimeLookUp = ref(92)
const focusRate = ref(88)
const abnormalStudents = ref(0)
const totalStudents = ref(0)
const waitPushReports = ref(0)

// 学生列表
const studentList = ref([])

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
  const userId = userInfo.id

  try {
    // 1. 获取老师的班级、科目
    const { data } = await axios.post('http://localhost:5001/teacher-class', {
      user_id: userId
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

// 跳转
const goToClass = () => alert('进入上课页面')
const goToMember = () => alert('班级成员')
const goToClassGroup = () => alert('班级信息')
const goToReport = () => alert('报告管理')
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
.stat-card.orange { border-left: 4px solid #faad14; }
.stat-card.red { border-left: 4px solid #f5222d; }
.stat-card.purple { border-left: 4px solid #722ed1; }
.stat-card.cyan { border-left: 4px solid #13c2c2; }

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

.block-header h3 {
  margin: 0;
  font-size: 16px;
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

.report-table {
  width: 100%;
  border-collapse: collapse;
}

.report-table th,
.report-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
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
</style>