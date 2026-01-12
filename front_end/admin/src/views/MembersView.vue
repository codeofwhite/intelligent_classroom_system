<template>
    <div class="members-container">
        <h3>会员管理</h3>
        <table class="members-table">
        <thead>
            <tr>
            <th>会员ID</th>
            <th>姓名</th>
            <th>注册日期</th>
            <th>状态</th>
            </tr>
        </thead>
        <tbody>
            <tr v-for="member in members" :key="member.id">
            <td>{{ member.id }}</td>
            <td>{{ member.name }}</td>
            <td>{{ member.registrationDate }}</td>
            <td>{{ member.status }}</td>
            </tr>
        </tbody>
        </table>
    </div>
</template>