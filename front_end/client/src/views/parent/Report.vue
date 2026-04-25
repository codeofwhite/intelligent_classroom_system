<template>
  <div class="report-page">
    <div class="header">
      <h2>👨‍👩‍👧‍👦 家长端 - 孩子课堂行为报告</h2>
      <p>课堂实时行为统计 & 学习状态分析</p>
    </div>

    <!-- 学生选择（预留） -->
    <div class="student-selector">
      <button 
        v-for="student in studentList" 
        :key="student.id" 
        class="student-btn"
        @click="loadStudentReport(student.id)"
      >
        {{ student.name }}
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
      请选择学生查看报告
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

// =========================================
// 学生列表（预留：后面你绑定 student_id + 姓名）
// =========================================
const studentList = ref([
  { id: 1, name: "张三" },
  { id: 2, name: "李四" },
  { id: 3, name: "王五" }
])

// 报告数据
const report = ref(null)

// =========================================
// ✅ 预留接口：加载【学生课堂报告】
// 后端接口：/api/student/report/:student_id
// =========================================
async function loadStudentReport(student_id) {
  try {
    const res = await axios.get(`http://localhost:5002/api/student/report/${student_id}`)
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