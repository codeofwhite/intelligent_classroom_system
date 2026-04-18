<template>
  <div class="members-container">
    <h3>会员管理</h3>

    <!-- 筛选区域：班级/小组筛选（基础行政管理功能） -->
    <div class="filter-bar">
      <div class="filter-item">
        <label>班级筛选：</label>
        <select v-model="searchClass">
          <option value="">全部班级</option>
          <option value="一年级1班">一年级1班</option>
          <option value="一年级2班">一年级2班</option>
          <option value="二年级1班">二年级1班</option>
        </select>
      </div>

      <div class="filter-item">
        <label>小组筛选：</label>
        <select v-model="searchGroup">
          <option value="">全部小组</option>
          <option value="A组">A组</option>
          <option value="B组">B组</option>
          <option value="C组">C组</option>
        </select>
      </div>
    </div>

    <!-- 会员表格 -->
    <table class="members-table">
      <thead>
        <tr>
          <th>会员ID</th>
          <th>姓名</th>
          <th>班级</th>
          <th>小组</th>
          <th>注册日期</th>
          <th>状态</th>
          <th>家长手机号（家校绑定）</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="member in filteredMembers" :key="member.id">
          <td>{{ member.id }}</td>
          <td>{{ member.name }}</td>
          <td>{{ member.className }}</td>
          <td>{{ member.groupName }}</td>
          <td>{{ member.registrationDate }}</td>
          <td :class="member.status === '正常' ? 'status-normal' : 'status-disable'">
            {{ member.status }}
          </td>
          <td>{{ member.parentPhone || '未绑定' }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// 模拟会员数据（包含班级、小组、家长手机号，满足行政管理需求）
const members = ref([
  {
    id: 1001,
    name: '张三',
    className: '一年级1班',
    groupName: 'A组',
    registrationDate: '2025-01-10',
    status: '正常',
    parentPhone: '13800138000' // 家校信息绑定
  },
  {
    id: 1002,
    name: '李四',
    className: '一年级1班',
    groupName: 'B组',
    registrationDate: '2025-01-11',
    status: '正常',
    parentPhone: '13900139000'
  },
  {
    id: 1003,
    name: '王五',
    className: '二年级1班',
    groupName: 'C组',
    registrationDate: '2025-01-12',
    status: '禁用',
    parentPhone: ''
  },
  {
    id: 1004,
    name: '赵六',
    className: '一年级2班',
    groupName: 'A组',
    registrationDate: '2025-01-13',
    status: '正常',
    parentPhone: '13700137000'
  }
])

// 班级/小组筛选条件
const searchClass = ref('')
const searchGroup = ref('')

// 计算属性：筛选后的会员列表
const filteredMembers = computed(() => {
  return members.value.filter(item => {
    // 班级筛选
    const matchClass = searchClass.value ? item.className === searchClass.value : true
    // 小组筛选
    const matchGroup = searchGroup.value ? item.groupName === searchGroup.value : true
    return matchClass && matchGroup
  })
})
</script>

<style scoped>
.members-container {
  width: 90%;
  margin: 30px auto;
  font-family: "Microsoft YaHei", sans-serif;
}

/* 筛选栏样式 */
.filter-bar {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 10px 15px;
  background: #f7f8fa;
  border-radius: 6px;
}

.filter-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-item select {
  padding: 5px 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  outline: none;
}

/* 表格样式 */
.members-table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.members-table th,
.members-table td {
  padding: 12px 15px;
  text-align: center;
  border: 1px solid #ebeef5;
}

.members-table th {
  background-color: #f0f2f5;
  font-weight: bold;
}

/* 状态样式 */
.status-normal {
  color: #67c23a;
  font-weight: bold;
}

.status-disable {
  color: #f56c6c;
  font-weight: bold;
}
</style>