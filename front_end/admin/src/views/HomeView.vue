<template>
  <div class="teacher-home">
    <h2>👋 您好，{{ teacherName }}！这是今天的教学概览。</h2>

    <div class="data-grid">
      <div class="stat-card">
        <p>今日课堂录像</p>
        <h3>4 个</h3>
        <span class="tag">已自动云同步</span>
      </div>
      <div class="stat-card">
        <p>待审阅报告</p>
        <h3 style="color: #ff4d4f;">12 份</h3>
        <span class="tag">学生课堂表现</span>
      </div>
      <div class="stat-card">
        <p>班级成员</p>
        <h3>45 人</h3>
        <span class="tag">在线 42 人</span>
      </div>
    </div>

    <div class="action-sections">
      <section class="action-block">
        <div class="block-header">
          <h3>📹 最近课堂录像</h3>
          <button class="text-btn">查看全部</button>
        </div>
        <div class="video-list">
          <div class="video-item" v-for="i in 2" :key="i">
            <div class="video-thumb">播放图标</div>
            <div class="video-info">
              <p class="title">数学：分数乘除法 (1)</p>
              <p class="time">2026-01-09 09:00 - 10:00</p>
            </div>
          </div>
        </div>
      </section>

      <section class="action-block">
        <div class="block-header">
          <h3>📝 待审阅行为报告</h3>
        </div>
        <table class="report-table">
          <thead>
            <tr>
              <th>学生</th>
              <th>评估项目</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>张小明</td>
              <td>课堂参与度</td>
              <td>10:15</td>
              <td><button class="action-btn">审阅</button></td>
            </tr>
            <tr>
              <td>李华</td>
              <td>专注力评分</td>
              <td>09:45</td>
              <td><button class="action-btn">审阅</button></td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>
</template>

<style scoped>
.data-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 24px 0;
}

.stat-card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.stat-card h3 {
  font-size: 28px;
  margin: 10px 0;
}

.action-sections {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.action-block {
  background: white;
  padding: 20px;
  border-radius: 8px;
  min-height: 300px;
}

.block-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

/* 模拟视频项 */
.video-item {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
  padding: 10px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
}

.video-thumb {
  width: 100px;
  height: 60px;
  background: #eee;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
}

/* 表格 */
.report-table {
  width: 100%;
  border-collapse: collapse;
}

.report-table th, .report-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

.action-btn {
  background: #1890ff;
  color: white;
  border: none;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
}
</style>