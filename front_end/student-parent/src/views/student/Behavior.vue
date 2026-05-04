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
          <strong>{{ reportList.length }} 节</strong>
        </div>
        <div>
          <span>平均专注度</span>
          <strong class="blue">{{ avgFocus }}%</strong>
        </div>
        <div>
          <span>总体评价</span>
          <strong class="green">{{ level }}</strong>
        </div>
      </div>
    </div>

    <!-- 历史课程报告列表 -->
    <div class="list-section">
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
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'

const reportList = ref([])
const currentDetail = ref(null)

// 读取用户信息
let user = null
try {
  user = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

// ✅ 正确字段：student_code
const student_code = user?.student_code

console.log("用户信息：", user)
console.log("学生CODE：", student_code)

// 平均专注度
const avgFocus = computed(() => {
  if (reportList.value.length === 0) return 0
  const sum = reportList.value.reduce((t, i) => t + (i.focus_rate || 0), 0)
  return Math.round(sum / reportList.value.length)
})

// 总体评价
const level = computed(() => {
  const avg = avgFocus.value
  if (avg >= 90) return '优秀'
  if (avg >= 80) return '良好'
  if (avg >= 70) return '一般'
  return '需努力'
})

// 加载我的报告
const loadMyReports = async () => {
  if (!student_code) {
    alert('未获取到学生信息：' + JSON.stringify(user))
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

// 打开详情
const openDetail = (item) => {
  currentDetail.value = item
}

// 关闭弹窗
const closeDetail = () => {
  currentDetail.value = null
}

onMounted(() => {
  loadMyReports()
})
</script>

<style scoped>
/* 整体布局 */
.report-page {
  padding: 24px;
  background: linear-gradient(to bottom, #f8faff, #eff4ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 头部标题 */
.header {
  text-align: center;
  margin-bottom: 24px;
}
.header h2 {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 6px 0;
  color: #2c3e50;
}
.header p {
  font-size: 14px;
  color: #7f8c8d;
  margin: 0;
}

/* 本周概况卡片 - 渐变精致款 */
.week-card {
  background: linear-gradient(to right, #4285f4, #64a6ff);
  border-radius: 18px;
  padding: 20px;
  margin-bottom: 24px;
  color: white;
  box-shadow: 0 8px 20px rgba(66, 133, 244, 0.25);
}
.week-card h3 {
  font-size: 17px;
  margin: 0 0 16px 0;
  font-weight: 500;
}
.items {
  display: flex;
  justify-content: space-around;
  text-align: center;
}
.items div {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.items span {
  font-size: 13px;
  opacity: 0.9;
}
.items strong {
  font-size: 18px;
  font-weight: 600;
}
.blue { color: #ffffff; }
.green { color: #d1fae5; }

/* 列表标题 */
.list-section h3 {
  font-size: 17px;
  font-weight: 600;
  margin: 0 0 14px 4px;
  color: #2c3e50;
}

/* 报告列表项 - 精致卡片 */
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
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
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
  color: #4285f4;
}
.arrow {
  color: #bdc3c7;
  font-size: 16px;
}

/* 弹窗遮罩 */
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
/* 弹窗内容 */
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

/* 弹窗头部 */
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

/* 详情内容 */
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

/* 统计数据卡片 */
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

/* 建议 & 评语卡片 */
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
</style>