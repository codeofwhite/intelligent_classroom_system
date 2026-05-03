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
.report-page {
  padding: 20px;
  background: #f7f8fa;
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

.header {
  text-align: center;
  margin-bottom: 16px;
}
.header h2 {
  font-size: 22px;
  margin: 0 0 4px 0;
}
.header p {
  font-size: 14px;
  color: #888;
  margin: 0;
}

.week-card {
  background: #fff;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 20px;
}
.week-card h3 {
  font-size: 16px;
  margin: 0 0 12px 0;
}
.items {
  display: flex;
  justify-content: space-around;
  text-align: center;
}
.items div {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.items span {
  font-size: 13px;
  color: #888;
}
.items strong {
  font-size: 16px;
  font-weight: bold;
}
.blue { color: #429dff; }
.green { color: #20c997; }

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

.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  z-index: 999;
}
.modal-content {
  background: #fff;
  width: 100%;
  max-width: 480px;
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
.modal-header h3 {
  margin: 0;
  font-size: 18px;
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
  font-size: 15px;
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
  display: block;
  margin-bottom: 4px;
}
.stats span {
  font-weight: bold;
  font-size: 16px;
}
.red { color: #f56c6c; }

.suggest, .comment {
  margin-bottom: 16px;
}
.suggest h4, .comment h4 {
  font-size: 15px;
  margin: 0 0 8px 0;
}
.suggest p, .comment p {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
}
</style>