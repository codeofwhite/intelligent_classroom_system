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

    <!-- 报告列表 -->
    <div class="list-section" v-if="reportList.length > 0">
      <h3>📋 课程报告记录</h3>
      <div
        class="item"
        v-for="item in reportList"
        :key="item.id"
        @click="openDetail(item)"
      >
        <div class="left">
          <div class="subject">课堂行为报告</div>
          <div class="date">{{ item.lesson_time }}</div>
        </div>
        <div class="right">
          <div class="score">{{ item.focus_rate }}%</div>
          <div class="arrow">></div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div class="empty-tip" v-else>
      请选择孩子查看课堂报告
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

          <div class="stats">
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
import { ref, onMounted } from 'vue'
import axios from 'axios'

let parentUser = null
try {
  parentUser = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

const studentList = ref([])
const selectedCode = ref(null)
const reportList = ref([])
const currentDetail = ref(null)

onMounted(async () => {
  if (!parentUser || parentUser.role !== 'parent') {
    alert('请以家长身份登录')
    return
  }

  try {
    const res = await axios.post('http://localhost:5001/parent-children', {
      user_code: parentUser.user_code
    })
    studentList.value = res.data.children
  } catch (err) {
    console.error(err)
    alert('获取孩子信息失败')
  }
})

async function loadStudentReport(student_code) {
  selectedCode.value = student_code
  try {
    const res = await axios.get('http://localhost:5002/api/student/my-reports', {
      params: { student_code }
    })
    reportList.value = res.data.list || []
  } catch (e) {
    alert("获取报告失败")
    console.error(e)
  }
}

const openDetail = (item) => {
  currentDetail.value = item
}

const closeDetail = () => {
  currentDetail.value = null
}
</script>

<style scoped>
.report-page {
  padding: 24px;
  background: linear-gradient(to bottom, #f9faff, #f1f5ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 头部 */
.header {
  text-align: center;
  margin-bottom: 24px;
}
.header h2 {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #2c3e50;
}
.header p {
  font-size: 14px;
  color: #7f8c8d;
  margin: 0;
}

/* 选择孩子 */
.student-selector {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-bottom: 26px;
  flex-wrap: wrap;
}
.student-btn {
  padding: 12px 20px;
  border: 1px solid #e2e8ff;
  background: #fff;
  border-radius: 12px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}
.student-btn.active {
  background: linear-gradient(90deg, #7c5fff, #9b77ff);
  color: white;
  border-color: #7c5fff;
  transform: scale(1.03);
}

/* 报告列表 */
.list-section h3 {
  font-size: 17px;
  font-weight: 600;
  margin: 0 0 14px 4px;
  color: #2c3e50;
}
.item {
  background: #ffffff;
  border-radius: 16px;
  padding: 18px 16px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: all 0.25s ease;
  border: 1px solid rgba(255,255,255,0.6);
}
.item:active {
  transform: scale(0.97);
}
.left .subject {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 6px;
  color: #2c3e50;
}
.left .date {
  font-size: 13px;
  color: #7f8c8d;
}
.right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.score {
  font-size: 18px;
  font-weight: bold;
  color: #7c5fff;
}
.arrow {
  color: #bdc3c7;
  font-size: 16px;
}

/* 弹窗 */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
  backdrop-filter: blur(4px);
}
.modal-content {
  background: #fff;
  width: 90%;
  max-width: 420px;
  border-radius: 22px;
  max-height: 85vh;
  overflow-y: auto;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
  animation: modalUp 0.3s ease;
}
@keyframes modalUp {
  from { transform: translateY(40px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.modal-header {
  padding: 20px;
  border-bottom: 1px solid #f1f1f1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.modal-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}
.modal-header button {
  background: #f5f6fa;
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  font-size: 18px;
  cursor: pointer;
  color: #7f8c8d;
  display: flex;
  align-items: center;
  justify-content: center;
}

.detail-body {
  padding: 24px;
}
.info-row {
  display: flex;
  justify-content: space-between;
  padding: 12px 0;
  font-size: 15px;
  color: #2c3e50;
}

/* 统计卡片 */
.stats {
  background: #f8f9fd;
  border-radius: 14px;
  padding: 18px 14px;
  display: flex;
  justify-content: space-around;
  margin: 20px 0;
}
.stats div {
  text-align: center;
}
.stats label {
  font-size: 13px;
  color: #7f8c8d;
  display: block;
  margin-bottom: 6px;
}
.stats span {
  font-weight: 600;
  font-size: 18px;
  color: #2c3e50;
}
.red { color: #ff4757 !important; }
.purple { color: #7c5fff !important; }

/* 建议 & 评语 */
.suggest, .comment {
  margin-bottom: 20px;
}
.suggest h4, .comment h4 {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 10px 0;
  color: #2c3e50;
}
.suggest p, .comment p {
  background: #f8f9fd;
  padding: 16px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.6;
  margin: 0;
  color: #34495e;
}

.empty-tip {
  text-align: center;
  color: #999;
  padding: 60px 20px;
  font-size: 15px;
}
</style>