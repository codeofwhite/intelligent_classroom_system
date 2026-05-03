<template>
  <div class="report-page">
    <div class="header">
      <h2>👨‍👩‍👧‍👦 家长端 - 孩子课堂行为报告</h2>
      <p>课堂实时行为统计 & 学习状态分析</p>
    </div>

    <div class="student-selector">
      <button 
        v-for="s in studentList" 
        :key="s.student_code" 
        class="student-btn"
        :class="{ active: selectedCode === s.student_code }"
        @click="loadStudentReport(s.student_code)"
      >
        {{ s.student_name }}
      </button>
    </div>

    <!-- 报告列表（和你参考版一模一样） -->
    <div class="list-section" v-if="reportList.length > 0">
      <h3>课程报告记录</h3>
      <div
        class="item"
        v-for="item in reportList"
        :key="item.id"
        @click="openDetail(item)"
      >
        <div class="left">
          <div class="subject">课堂报告</div>
          <div class="date">{{ item.lesson_time }} </div>
        </div>
        <div class="right">
          <div class="score">{{ item.focus_rate }}%</div>
          <div class="arrow">></div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div class="empty-tip" v-else>
      请选择孩子查看报告
    </div>

    <!-- 详情弹窗（完全照搬你能用的版本） -->
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
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// 读取家长登录信息
let parentUser = null
try {
  parentUser = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

const studentList = ref([])
const selectedCode = ref(null)
const reportList = ref([])        // 报告列表
const currentDetail = ref(null)   // 弹窗详情

// 获取孩子列表
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

// ✅ 加载孩子报告（完全和你参考版一样，不会404）
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

// 打开弹窗
const openDetail = (item) => {
  currentDetail.value = item
}

// 关闭弹窗
const closeDetail = () => {
  currentDetail.value = null
}
</script>

<style scoped>
/* 完全照搬你能用的样式 */
.report-page {
  padding: 20px;
  background: #f7f8fa;
  min-height: 100vh;
}
.header {
  text-align: center;
  margin-bottom: 16px;
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

/* 列表样式和你参考版完全一样 */
.list-section h3 {
  font-size: 16px;
  margin: 0 0 10px 4px;
}
.item {
  background: #fff;
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.left .subject {
  font-weight: bold;
  font-size: 15px;
  margin-bottom: 4px;
}
.left .date {
  font-size: 13px;
  color: #999;
}
.right {
  display: flex;
  align-items: center;
  gap: 8px;
}
.score {
  font-size: 16px;
  font-weight: bold;
  color: #429dff;
}
.arrow {
  color: #ccc;
  font-size: 16px;
}

/* 弹窗 */
.modal {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  z-index: 999;
}
.modal-content {
  background: #fff;
  width: 100%; max-width: 480px;
  border-radius: 20px 20px 0 0;
  max-height: 80vh;
  overflow-y: auto;
}
.modal-header {
  padding: 16px;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.modal-header button {
  background: none;
  border: none;
  font-size: 22px;
  cursor: pointer;
}
.detail-body {
  padding: 20px;
}
.info-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
}
.stats {
  background: #f5f7fa;
  border-radius: 10px;
  padding: 14px;
  display: flex;
  justify-content: space-around;
  margin: 16px 0;
}
.stats div {
  text-align: center;
}
.stats label {
  font-size: 13px;
  color: #888;
}
.red { color: #f56c6c; }
.blue { color: #429dff; }

.empty-tip {
  text-align: center;
  color: #999;
  padding: 40px;
}
</style>