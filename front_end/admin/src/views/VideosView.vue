<template>
  <div class="monitor-container">
    <h3>实时监控画面 (AI 模式)</h3>
    <div class="video-window">
      <img 
        src="http://localhost:5000/video_feed" 
        alt="视频流加载中..." 
        style="width: 100%; border-radius: 8px;"
      />
    </div>
    <div class="status-panel">
      <p>状态：<span style="color: green">● 正在实时分析</span></p>
      <p>当前检测到：<strong v-if="alert">有人低头！</strong></p>
    </div>
  </div>
</template>