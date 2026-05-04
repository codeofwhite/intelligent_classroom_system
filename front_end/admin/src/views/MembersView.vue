<template>
  <div class="member-wrapper">
    <div class="member-card">
      <div class="card-header">
        <h3>👨‍🎓 班级成员管理</h3>
        <div class="class-info">
          当前班级：<span>{{ className }}</span> · 学生总数：<span>{{ studentList.length }}</span>
        </div>
      </div>

      <div class="table-box">
        <table class="member-table">
          <thead>
            <tr>
              <th>序号</th>
              <th>学生姓名</th>
              <th>性别</th>
              <th>班级</th>
              <th>绑定家长</th>
              <th>家长电话</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(s, index) in studentList" :key="s.student_code">
              <td>{{ index + 1 }}</td>
              <td class="student-name">{{ s.student_name }}</td>
              <td>{{ s.gender }}</td>
              <td>{{ className }}</td>
              <td>{{ s.parent_name || '未绑定' }}</td>
              <td>{{ s.parent_phone || '—' }}</td>
              <td>
                <span class="status-active">正常</span>
              </td>
            </tr>
          </tbody>
        </table>

        <!-- 空数据提示 -->
        <div class="empty" v-if="studentList.length === 0">
          🧑‍🎓 暂无学生数据
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const className = ref('')
const studentList = ref([])

const loadData = async () => {
  const userInfo = JSON.parse(localStorage.getItem('userInfo'))
  if (!userInfo) return

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-students', {
      user_code: userInfo.user_code
    })
    className.value = data.class_name
    studentList.value = data.students
  } catch (err) {
    console.error(err)
  }
}

onMounted(() => loadData())
</script>

<style scoped>
/* 外层布局 */
.member-wrapper {
  width: 100%;
  padding: 24px;
  box-sizing: border-box;
  background: #f5f7fa;
  min-height: 100vh;
}

/* 卡片容器 */
.member-card {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

/* 卡片头部 */
.card-header {
  padding: 20px 24px;
  border-bottom: 1px solid #f0f2f5;
}

.card-header h3 {
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1d2129;
}

.class-info {
  font-size: 14px;
  color: #4e5969;
}

.class-info span {
  color: #1677ff;
  font-weight: 500;
}

/* 表格区域 */
.table-box {
  padding: 16px 24px 24px;
}

.member-table {
  width: 100%;
  border-collapse: collapse;
}

.member-table thead th {
  text-align: left;
  padding: 14px 12px;
  font-size: 14px;
  font-weight: 600;
  color: #4e5969;
  background: #fafbfc;
  border-bottom: 1px solid #e5e6eb;
}

.member-table tbody td {
  padding: 14px 12px;
  font-size: 14px;
  color: #1d2129;
  border-bottom: 1px solid #f0f2f5;
}

.member-table tbody tr:hover {
  background: #fafbfc;
  transition: background 0.2s;
}

.student-name {
  font-weight: 500;
  color: #1677ff;
}

/* 状态标签 */
.status-active {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  color: #00b42a;
  background: #e6ffed;
  border: 1px solid #b7eb8f;
}

/* 空状态 */
.empty {
  text-align: center;
  padding: 60px 0;
  font-size: 14px;
  color: #86909c;
}
</style>