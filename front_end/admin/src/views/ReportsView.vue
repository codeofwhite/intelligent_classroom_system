<template>
  <div class="student-report-page">
    <h3>学生行为分析报告 · 批改页面</h3>

    <!-- 学生基础信息 -->
    <div class="student-info">
      <p>姓名：张三</p>
      <p>班级：一年级1班</p>
      <p>课程：数学 · 分数乘除法</p>
      <p>日期：2025-12-19</p>
    </div>

    <!-- 行为分析图表/数据 -->
    <div class="behavior-data">
      <p>抬头率：94%</p>
      <p>专注度：90%</p>
      <p>举手次数：3次</p>
      <p>低头/走神：1次</p>
    </div>

    <!-- 老师批改区域 -->
    <div class="teacher-edit">
      <label>老师评分：</label>
      <input type="number" placeholder="1-100分" />

      <label>评语：</label>
      <textarea placeholder="输入学生课堂表现评语"></textarea>

      <button>保存报告</button>
      <button>推送给家长</button>
    </div>
  </div>
</template>