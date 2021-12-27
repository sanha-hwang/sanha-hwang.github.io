---
layout: post
title:  "Operation Systems_1강 Overview"
date: 2021-12-27 06:00:00
category: CS_Operating Systems
use_math: true
tags: OS
---

#### 이 글을 시작하며

본 게시물은 CS(computer science)를 공부하기 위한 첫번째 과목으로 "Operating Systems"를 다루며 나만의 커리큘럼을 만들어가고자 포스팅을 시작한다. 강의는 "Berkeley CS 162 ver2013"을 바탕으로 참고교재로는 "Operating System Concepts"(저자: Abraham Silberschatz, 일명 공룡책), "Operating systems : three easy pieces"(저자 : Remzi H. Arpaci-Dusseau, Andrea C. Arpaci-Dusseau)를 참고하였다.

## Computer system operation

(그림 첨부)

## challenge : complexity
다양한 시스템과 디바이스, 다른 하드웨어 구조, 어플리케이션 사이의 경쟁, 예측하지 못한 방향에서의 실패, 다양한 공격 등에 의해 OS의 구성은 점점 더 복잡해지고 있다. 어떻게 효율적으로 컴퓨터 전반적으로 통솔하는 OS를 살필 지 앞으로 배워나가본다.

## Virtual machines

software는 추상적 장치로 보일 수 있도록 한다. 자신만의 machine, hardware가 우리가 원하는 특징을 가진 것 처럼 보이게 한다. 즉 OS는 사용자가 보기에 ~인 척 하는 놈이다. 여러 개의 CPU가 존재하는 척, 별도의 RAM을 가지는 척 등 이 것들을 가상화(virtualization)이라고 한다.

### system VM 
system VM : 물리적인 시스템 OS위에 논리적인 가상 OS를 올려서 독립적인 동작이 가능하도록 하는 시스템 차원의 VM
(ex. VMware, Fusion, Parallels)<br>

OS 충돌이 일어났을 때, 하나의 vm을 제한한다. 여러 테스트 프로그램을 다양한 OS애서 테스트가 가능하다.

### process VM
process VM : 운영 체제 안에서 일반 응용 프로그램을 돌리고 단일 프로세스를 지원한다. (ex. java)<br>

각각 memory, CPU time, devices를 차지하며, Device 간 상호 작용을 활용한다. process끼지 서로 독립적이며, 상호작용이 안전하고 안정적이다.

## An OS is simillar to a goverment

OS as a Traffic Cop
- Manage all resources
- settle conflicting request
- prevent errors and improper use of the computer

OS as a facilitator
- provides facilities/services that everyone needs
- standard libraries, windowing systems
- make application programming easier, faster, less error

#### 이 글을 마치며

영어 강의라 전부 다 이해하기에는 실력이 부족하였다. 하지만 어느 대학교 강의 첫 강과 같이  교수 소개, 과제 소개 등등 설명과 함께 칩의 역사 및 OS 하드웨어의 발전 등을 오리엔테이션하였다. <br>
"Operating systems : three easy pieces"에서도 OS의 가장 중요한 세 가지 개념을 보자고 하였다. 가상화(virtualization), 병행성(concurrency), 영속성(persistence) 세 가지를 통해 OS를 살펴보려한다.<br>
완강할 수 있도록 노력하겠다.

### 전체 진도율 : 1/24
