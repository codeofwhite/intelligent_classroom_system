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
          <strong>5 节</strong>
        </div>
        <div>
          <span>平均专注度</span>
          <strong class="blue">88%</strong>
        </div>
        <div>
          <span>总体评价</span>
          <strong class="green">良好</strong>
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
          <div class="subject">{{ item.subject }}</div>
          <div class="date">{{ item.date }} · {{ item.lesson }}</div>
        </div>
        <div class="right">
          <div class="score">{{ item.focus }}%</div>
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
            <span>课程</span>
            <span>{{ currentDetail.subject }}</span>
          </div>
          <div class="info-row">
            <span>时间</span>
            <span>{{ currentDetail.date }} {{ currentDetail.lesson }}</span>
          </div>
          <div class="info-row">
            <span>专注度</span>
            <span class="blue">{{ currentDetail.focus }}%</span>
          </div>

          <div class="stats">
            <div>
              <label>抬头次数</label>
              <span>{{ currentDetail.lookUp }}</span>
            </div>
            <div>
              <label>举手次数</label>
              <span>{{ currentDetail.handUp }}</span>
            </div>
            <div>
              <label>走神次数</label>
              <span class="red">{{ currentDetail.disturb }}</span>
            </div>
          </div>

          <!-- AI 建议 -->
          <div class="suggest">
            <h4>💡 AI 学习建议</h4>
            <p>{{ currentDetail.suggest }}</p>
          </div>

          <!-- 老师评语 -->
          <div class="comment" v-if="currentDetail.comment">
            <h4>👨‍🏫 老师评语</h4>
            <p>{{ currentDetail.comment }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// 课程报告列表
const reportList = ref([
  {
    id: 1,
    subject: '数学',
    date: '2026-04-18',
    lesson: '第3节',
    focus: 92,
    lookUp: 18,
    handUp: 2,
    disturb: 1,
    suggest: '本节课整体专注度很高，偶尔一次走神，下次可以尽量保持坐姿哦！',
    comment: '课堂表现很棒，继续保持！'
  },
  {
    id: 2,
    subject: '语文',
    date: '2026-04-17',
    lesson: '第5节',
    focus: 85,
    lookUp: 14,
    handUp: 1,
    disturb: 3,
    suggest: '后半节课注意力有些下降，可以尝试跟着老师思路小声跟读。',
    comment: '积极思考，回答问题声音再大一点就更好了。'
  },
  {
    id: 3,
    subject: '英语',
    date: '2026-04-16',
    lesson: '第2节',
    focus: 90,
    lookUp: 16,
    handUp: 3,
    disturb: 0,
    suggest: '全程非常专注，积极互动，是非常棒的课堂状态！',
    comment: '表现优秀，值得表扬！'
  }
])

const currentDetail = ref(null)

const openDetail = (item) => {
  currentDetail.value = item
}

const closeDetail = () => {
  currentDetail.value = null
}
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

/* 本周概况 */
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

/* 报告列表 */
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