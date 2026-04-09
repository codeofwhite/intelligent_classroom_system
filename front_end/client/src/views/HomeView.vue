<template>
  <div class="home-view">
    <header class="welcome-section">
      <div class="user-info">
        <div class="avatar">🌟</div>
        <div class="text">
          <h2>早安，小探险家！</h2>
          <p>今天也是充满进步的一天，加油！</p>
        </div>
      </div>
      <div class="stats-badge">
        <span>已连续学习 <strong>5</strong> 天</span>
      </div>
    </header>

    <section class="main-actions">
      <div class="action-card primary">
        <div class="icon">🚀</div>
        <h3>进入直播课堂</h3>
        <p>王老师的数学课正在进行中...</p>
      </div>
      <div class="action-card secondary">
        <div class="icon">📚</div>
        <h3>我的课程表</h3>
        <p>查看本周安排</p>
      </div>
    </section>

    <section class="learning-progress">
      <div class="section-header">
        <h3>今日学习任务</h3>
        <span class="more">查看全部 ></span>
      </div>
      <div class="task-list">
        <div class="task-item completed">
          <input type="checkbox" checked disabled>
          <span>[语文] 朗读课文《秋天》</span>
          <span class="status">已完成</span>
        </div>
        <div class="task-item">
          <input type="checkbox">
          <span>[数学] 完成分数运算练习题</span>
          <span class="status">待做</span>
        </div>
      </div>
    </section>

    <section class="honor-roll">
      <div class="honor-card">
        <p>获得的星钻：💎 120</p>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: 70%"></div>
        </div>
        <small>再攒 30 颗星钻即可兑换新挂件！</small>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home-view {
  padding: 0;
  color: #333;
  min-height: 100%;
}

.welcome-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #fff;
  padding: 20px 24px;
  border-radius: 16px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 14px;
}

.avatar {
  font-size: 30px;
  width: 50px;
  height: 50px;
  background: #f0f7ff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.user-info h2 {
  margin: 0 0 4px;
  font-size: 18px;
}

.user-info p {
  margin: 0;
  font-size: 13px;
  color: #666;
}

.stats-badge {
  font-size: 14px;
  color: #4a90e2;
  background: #e8f3ff;
  padding: 8px 14px;
  border-radius: 30px;
}

.main-actions {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 26px;
}

.action-card {
  border-radius: 16px;
  padding: 22px 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s;
}

.action-card:hover {
  transform: translateY(-2px);
}

.action-card.primary {
  background: linear-gradient(135deg, #6ebdfa, #4a90e2);
  color: white;
}

.action-card.secondary {
  background: white;
  border: 1px solid #e8f3ff;
}

.action-card .icon {
  font-size: 26px;
  margin-bottom: 10px;
}

.action-card h3 {
  margin: 0 0 6px;
  font-size: 16px;
}

.action-card p {
  margin: 0;
  font-size: 13px;
  opacity: 0.9;
}

.learning-progress {
  margin-bottom: 26px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.section-header h3 {
  margin: 0;
  font-size: 16px;
}

.more {
  font-size: 13px;
  color: #4a90e2;
  cursor: pointer;
}

.task-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: white;
  padding: 16px;
  border-radius: 12px;
  margin-bottom: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.task-item.completed {
  background: #f0fff4;
}

.status {
  margin-left: auto;
  font-size: 12px;
  color: #888;
}

.completed .status {
  color: #28a745;
}

.honor-card {
  background: white;
  padding: 22px;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.honor-card p {
  margin: 0 0 12px;
  font-weight: 500;
}

.progress-bar-container {
  width: 100%;
  height: 8px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(to right, #ffd43b, #ffb100);
  border-radius: 10px;
}

.honor-card small {
  color: #888;
  font-size: 12px;
}
</style>