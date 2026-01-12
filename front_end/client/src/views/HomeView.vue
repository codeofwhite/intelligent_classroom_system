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
  padding: 20px;
  background-color: #f0f7ff; 
  color: #333;
  min-height: 100vh;
}

.action-card {
  background: white;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  margin-bottom: 15px;
  border-bottom: 4px solid #e0e0e0;
}

.primary {
  background: linear-gradient(135deg, #6ebdfa, #4a90e2);
  color: white;
}

.task-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: white;
  padding: 12px;
  border-radius: 10px;
  margin-bottom: 8px;
}
</style>