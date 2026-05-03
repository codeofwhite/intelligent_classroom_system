<template>
  <div class="report-page">
    <div class="header">
      <h2>👨‍👩‍👧‍👦 家长端 - 孩子课堂行为报告</h2>
      <p>课堂实时行为统计 & 学习状态分析</p>
    </div>

    <!-- 自动加载孩子列表 -->
    <div class="student-selector">
      <button 
        v-for="s in studentList" 
        :key="s.student_id" 
        class="student-btn"
        :class="{ active: selectedStudentId === s.student_id }"
        @click="loadStudentReport(s.student_id)"
      >
        {{ s.student_name }}
      </button>
    </div>

    <!-- 报告内容 -->
    <div v-if="report" class="report-card">
      <h3>📊 课堂行为统计报告</h3>

      <div class="stats-grid">
        <div class="stat-item" v-for="(val, key) in report.behavior_counts" :key="key">
          <div class="label">{{ key }}</div>
          <div class="count">{{ val }} 次</div>
        </div>
      </div>

      <div class="info">
        <p>总分析帧数：{{ report.total_frames }}</p>
        <p>生成时间：{{ report.analyzed_time }}</p>
      </div>
    </div>

    <div v-else class="empty-tip">
      请选择孩子查看报告
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// 当前登录家长信息
let parentUser = null
try {
  parentUser = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

// 孩子列表（自动加载）
const studentList = ref([])
const selectedStudentId = ref(null)
const report = ref(null)

// =========================================
// 1. 进入页面自动加载【我的孩子】
// =========================================
onMounted(async () => {
  if (!parentUser || parentUser.role !== 'parent') {
    alert('请以家长身份登录')
    return
  }

  try {
    const res = await axios.post('http://localhost:5001/parent-children', {
      user_id: parentUser.id
    })
    studentList.value = res.data.children
  } catch (err) {
    console.error(err)
    alert('获取孩子信息失败')
  }
})

// =========================================
// 2. 加载孩子课堂报告
// =========================================
async function loadStudentReport(student_id) {
  selectedStudentId.value = student_id
  try {
    const res = await axios.get(`http://localhost:5002/api/student/my-reports/${student_id}`)
    report.value = res.data
  } catch (e) {
    alert("获取报告失败")
    console.error(e)
  }
}
</script>

<style scoped>
.report-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 30px 20px;
}
.header {
  text-align: center;
  margin-bottom: 30px;
}
.student-selector {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-bottom: 30px;
}
.student-btn {
  padding: 10px 18px;
  border: 1px solid #42b983;
  background: #fff;
  border-radius: 8px;
  cursor: pointer;
}
.student-btn.active {
  background: #42b983;
  color: white;
}
.report-card {
  background: #fff;
  padding: 24px;
  border-radius: 16px;
  box-shadow: 0 2px 10px #00000008;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin: 20px 0;
}
.stat-item {
  background: #f9fafb;
  padding: 16px;
  border-radius: 10px;
  text-align: center;
}
.count {
  font-size: 20px;
  font-weight: bold;
  margin-top: 8px;
}
.empty-tip {
  text-align: center;
  color: #999;
  padding: 40px;
}
</style>