#!/usr/bin/env python3
"""Build matching editable draw.io and SVG concept figures (standard library only)."""

from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/assets/concepts/diagrams"
INK, MUTED = "#20364C", "#52677C"
BLUE, GREEN, ORANGE, PURPLE = "#2475A5", "#29836A", "#B87623", "#7958A5"
PALE = {BLUE: "#EDF6FC", GREEN: "#EDF8F3", ORANGE: "#FFF6E8", PURPLE: "#F5F0FA", MUTED: "#F4F7FA"}


class Figure:
    def __init__(self, name, locale, title, subtitle, height=600):
        self.name, self.locale, self.width, self.height = name, locale, 1100, height
        self.svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="1100", height=str(height), viewBox=f"0 0 1100 {height}", role="img", **{"aria-labelledby": "title desc"})
        ET.SubElement(self.svg, "title", id="title").text = title
        ET.SubElement(self.svg, "desc", id="desc").text = subtitle
        defs = ET.SubElement(self.svg, "defs")
        for color in [BLUE, GREEN, ORANGE, PURPLE, MUTED]:
            marker = ET.SubElement(defs, "marker", id=color[1:], markerWidth="9", markerHeight="9", refX="8", refY="4.5", orient="auto", markerUnits="userSpaceOnUse")
            ET.SubElement(marker, "path", d="M0 0 L9 4.5 L0 9 Z", fill=color)
        self.mx = ET.Element("mxfile", host="app.diagrams.net", version="31.4.5", type="device")
        diagram = ET.SubElement(self.mx, "diagram", id=f"{name}-{locale}", name=title)
        model = ET.SubElement(diagram, "mxGraphModel", dx="1100", dy=str(height), grid="1", gridSize="10", page="1", pageScale="1", pageWidth="1100", pageHeight=str(height), background="#FFFFFF")
        self.cells = ET.SubElement(model, "root")
        ET.SubElement(self.cells, "mxCell", id="0")
        ET.SubElement(self.cells, "mxCell", id="1", parent="0")
        self.counter = 2
        self.rect(0, 0, 1100, height, fill="#FFFFFF", stroke="none", rounded=False)
        self.text(36, 20, 1028, 40, title, size=29, bold=True, align="left")
        self.text(36, 65, 1028, 38, subtitle, size=18, color=MUTED, align="left")

    def cell(self, value, style, x, y, w, h):
        c = ET.SubElement(self.cells, "mxCell", id=str(self.counter), value=value, style=style, vertex="1", parent="1")
        self.counter += 1
        ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=BLUE, rounded=True, ellipse=False):
        if ellipse:
            ET.SubElement(self.svg, "ellipse", cx=str(x+w/2), cy=str(y+h/2), rx=str(w/2), ry=str(h/2), fill=fill, stroke=stroke, **{"stroke-width": "2"})
        else:
            ET.SubElement(self.svg, "rect", x=str(x), y=str(y), width=str(w), height=str(h), rx="12" if rounded else "0", fill=fill, stroke=stroke, **{"stroke-width": "2"})
        self.cell("", f"{'ellipse;' if ellipse else ''}rounded={int(rounded)};arcSize=12;fillColor={fill};strokeColor={stroke};strokeWidth=2;", x,y,w,h)

    def text(self, x, y, w, h, value, size=21, bold=False, color=INK, align="center"):
        lines = value.split("\n")
        line_height = size * 1.38
        tx = x + w/2 if align == "center" else x
        sy = y + (h - line_height*len(lines))/2 + size
        t = ET.SubElement(self.svg, "text", x=str(tx), y=str(sy), fill=color, **{"font-family": "Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif", "font-size": str(size), "font-weight": "700" if bold else "400", "text-anchor": "middle" if align == "center" else "start"})
        for i, line in enumerate(lines):
            ET.SubElement(t, "tspan", x=str(tx), dy="0" if i == 0 else str(line_height)).text = line
        self.cell(value, f"text;html=0;whiteSpace=wrap;fillColor=none;strokeColor=none;align={align};verticalAlign=middle;fontFamily=Microsoft YaHei;fontSize={size};fontStyle={int(bold)};fontColor={color};spacing=0;",x,y,w,h)

    def box(self, x,y,w,h,heading,body="",color=BLUE):
        self.rect(x,y,w,h,PALE[color],color)
        self.text(x+12,y+9,w-24,34,heading,22,True)
        if body:
            self.text(x+12,y+47,w-24,h-56,body,18,color=MUTED)

    def line(self, points, color=BLUE, arrow=True, dash=False, width=2.5):
        attrs={"points": " ".join(f"{x},{y}" for x,y in points), "fill":"none", "stroke":color,"stroke-width":str(width),"stroke-linejoin":"round"}
        if arrow: attrs["marker-end"]=f"url(#{color[1:]})"
        if dash: attrs["stroke-dasharray"]="7 5"
        ET.SubElement(self.svg,"polyline",attrs)
        c=ET.SubElement(self.cells,"mxCell",id=str(self.counter),value="",edge="1",parent="1",style=f"edgeStyle=none;rounded=0;html=0;strokeColor={color};strokeWidth={width};endArrow={'block' if arrow else 'none'};endFill=1;dashed={int(dash)};")
        self.counter+=1
        g=ET.SubElement(c,"mxGeometry",relative="1",**{"as":"geometry"})
        ET.SubElement(g,"mxPoint",x=str(points[0][0]),y=str(points[0][1]),**{"as":"sourcePoint"})
        ET.SubElement(g,"mxPoint",x=str(points[-1][0]),y=str(points[-1][1]),**{"as":"targetPoint"})
        if len(points)>2:
            a=ET.SubElement(g,"Array",**{"as":"points"})
            for x,y in points[1:-1]: ET.SubElement(a,"mxPoint",x=str(x),y=str(y))

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        for suffix, root in [("svg", self.svg),("drawio",self.mx)]:
            ET.indent(root)
            ET.ElementTree(root).write(OUT/f"{self.name}.{self.locale}.{suffix}",encoding="utf-8",xml_declaration=True)


def generate(locale):
    def t(zh,en): return zh if locale=="zh-CN" else en
    f=Figure("architecture",locale,t("从数据到导航闭环","From data to the navigation loop"),t("蓝色：环境运行　绿色：离线训练　紫色：模型接口","Blue: environment   Green: offline training   Purple: policy interface"),650)
    f.box(40,130,260,120,"Episode",t("指令 · 起点 · 目标\nreference_path · scene_id","Instruction · start · goal\nreference_path · scene_id"))
    f.box(40,300,260,105,"GeoTIFF",t("带地理坐标的卫星影像","Georeferenced imagery"))
    f.box(390,130,300,120,"Env + VLNTask",t("Episode 生命周期\n传感器 · 动作 · 指标","Episode lifecycle\nSensors · actions · measures"))
    f.box(390,300,300,105,"SatSimWrapper → SatSim",t("更新位姿 · 渲染 RGB","Update pose · render RGB"))
    f.box(800,130,260,120,"PolicyAdapter",t("预处理 · 历史 · 动作队列\n模型推理","Preprocessing · history\nInference · action queue"),PURPLE)
    f.box(800,300,260,105,"Evaluator",t("选择 Episode · 驱动闭环\n保存与汇总结果","Select episodes · run loop\nWrite and aggregate results"),PURPLE)
    f.line([(300,190),(390,190)]); f.line([(300,350),(390,350)]);f.line([(540,250),(540,300)])
    f.line([(690,165),(800,165)]);f.text(691,120,108,35,"obs",16)
    f.line([(800,218),(690,218)],PURPLE);f.text(691,222,108,30,"action",16)
    f.line([(930,300),(930,250)],PURPLE)
    f.line([(800,350),(735,350),(735,280),(660,280),(660,250)],PURPLE,dash=True)
    f.box(40,500,310,100,t("专家路径跟随","Expert path following"),t("Episode + SatSim → RGB / 动作","Episode + SatSim → RGB / actions"),GREEN)
    f.box(410,500,280,100,t("离线训练","Offline training"),"Classic / VLM",GREEN)
    f.box(800,500,260,100,"Checkpoint",t("供 PolicyAdapter 加载","Loaded by PolicyAdapter"),GREEN)
    f.line([(350,550),(410,550)],GREEN);f.line([(690,550),(800,550)],GREEN)
    f.save()

    f=Figure("camera-geometry",locale,t("高度与视场角决定地面覆盖","Altitude and field of view set ground coverage"),t("平面俯视模型；右侧展示 GeoTIFF 到 RGB 的实际处理顺序","Planar downward-looking model; rendering stages are shown on the right"),580)
    f.line([(80,425),(500,425)],MUTED,False)
    f.line([(290,145),(105,425),(475,425),(290,145)],BLUE,False)
    f.line([(290,145),(290,425)],MUTED,False,True)
    f.rect(277,132,26,26,PALE[BLUE],BLUE,ellipse=True)
    # Put labels outside the viewing cone, with separate angle and height guides.
    f.line([(257,195),(268,202),(279,206),(290,208),(301,206),(312,202),(323,195)],BLUE,False)
    f.text(60,173,155,44,"HFOV = α",22)
    f.line([(215,195),(242,195),(257,195)],BLUE,False)
    f.line([(310,145),(530,145)],MUTED,False,True)
    f.line([(475,425),(530,425)],MUTED,False,True)
    f.line([(520,285),(520,145)],MUTED)
    f.line([(520,285),(520,425)],MUTED)
    f.text(535,245,75,80,t("高度\nh","Altitude\nh"),18,color=MUTED)
    f.line([(105,459),(475,459)],BLUE)
    f.text(60,466,470,50,"Lx = 2h tan(α / 2)",23,True)
    f.box(620,125,420,80,t("1  计算覆盖与旋转后的边界","1  Compute footprint and bounds"),"",BLUE)
    f.box(620,235,420,80,t("2  读取扩大后的影像窗口","2  Read an expanded raster window"),"",BLUE)
    f.box(620,345,420,80,t("3  按 heading 旋转、中心裁剪","3  Rotate by heading, center-crop"),"",BLUE)
    f.box(620,455,420,80,t("4  缩放为 W × H RGB","4  Resize to W × H RGB"),"",GREEN)
    for y in [205,315,425]: f.line([(830,y),(830,y+30)])
    f.save()

    f=Figure("coordinates",locale,t("地理坐标、像素坐标与相对位姿","Geographic coordinates, pixels and relative pose"),t("heading 从正北顺时针增加；相对位移坐标轴固定在 Episode 起始朝向","Heading increases clockwise from north; relative axes stay fixed to the initial heading"),630)
    f.box(35,125,310,105,"WGS84",t("经度 λ · 纬度 φ · 高度 h\n公共 position 接口","Longitude λ · latitude φ · h\nPublic position interface"))
    f.box(395,125,310,105,"EPSG:3857",t("投影 x / y\n移动与渲染窗口","Projected x / y\nMovement and raster windows"))
    f.box(755,125,310,105,t("影像像素","Raster pixels"),t("col 向右 · row 向下\nGeoTIFF affine transform","col right · row down\nGeoTIFF affine transform"))
    f.line([(345,178),(395,178)]);f.line([(705,178),(755,178)])
    f.line([(170,520),(170,300)],MUTED);f.text(110,260,120,40,"N / 0°",21)
    f.line([(170,520),(370,520)],MUTED);f.text(340,530,110,40,"E / 90°",21)
    f.line([(170,520),(325,365)],BLUE);f.text(290,310,170,45,"forward / θ₀",21,color=BLUE)
    f.line([(170,520),(255,605)],GREEN);f.text(265,573,180,40,"right / θ₀ + 90°",19,color=GREEN)
    f.rect(161,511,18,18,BLUE,BLUE,ellipse=True)
    f.text(45,527,113,55,t("Episode 起点","Episode start"),17)
    f.box(540,290,500,145,"agent_pose",t("沿起始 forward / right 的位移\nsin(Δheading), cos(Δheading)\nreset → [0, 0, 0, 1]","Displacement along initial forward / right\nsin(Δheading), cos(Δheading)\nreset → [0, 0, 0, 1]"),GREEN)
    f.box(540,475,500,105,t("动作更新","Action updates"),"LEFT: θ − 15°     RIGHT: θ + 15°\nFORWARD: 10 m",PURPLE)
    f.save()

    f=Figure("episode-loop",locale,t("一次 Episode 的闭环","One episode, one feedback loop"),t("每次 act 返回一个基础动作；模型可以在内部维护多动作队列","Each act returns one primitive action; an adapter may maintain an action queue"),660)
    f.box(40,130,295,110,t("初始化 Episode","Initialize episode"),"Env.reset_to_episode()\nPolicyAdapter.reset()")
    f.box(410,130,280,110,t("当前观测","Current observation"),"RGB · instruction · agent_pose")
    f.box(770,130,285,110,"PolicyAdapter.act()",t("查询模型或取出队列动作","Query model or pop action"),PURPLE)
    f.line([(335,185),(410,185)]);f.line([(690,185),(770,185)])
    f.box(770,310,285,110,"Env.step(action)",t("SatSim 更新状态并渲染\nTask 更新传感器与指标","SatSim: update state and RGB\nTask: sensors and measures"))
    f.line([(912,240),(912,310)],PURPLE)
    f.box(410,310,280,110,t("判断终止","Check termination"),"STOP / max_steps",ORANGE)
    f.line([(770,365),(690,365)])
    f.line([(550,310),(550,240)],GREEN);f.text(390,258,150,35,t("继续：新观测","Continue: new obs"),17,color=GREEN)
    f.box(410,505,280,100,t("记录结果","Write result"),"metrics + action trace",GREEN)
    f.line([(550,420),(550,505)],ORANGE);f.text(565,446,175,35,t("结束","Finished"),18,color=ORANGE,align="left")
    f.text(40,330,285,155,t("Env 管理步数与终止\nEvaluator 保存结果\n下一 Episode 重置模型历史","Env tracks steps and termination\nEvaluator persists results\nNext episode resets policy history"),19,color=MUTED)
    f.save()

    f=Figure("expert-flow",locale,t("把参考路径转换为专家动作","Turn a reference path into expert actions"),t("中间路径点到达后继续跟随；最终目标到达后执行 STOP","Advance after intermediate waypoints; execute STOP at the final goal"),660)
    f.box(35,130,295,110,"reference_path[1:]",t("必要时追加最终 goal\n按顺序选择目标点","Append final goal when needed\nSelect targets in order"))
    f.box(405,130,295,110,t("距离 d 与朝向误差 Δθ","Distance + heading error"),t("读取当前位姿\n计算目标方位角","Read current pose\nCompute bearing to target"))
    f.box(775,130,290,110,t("到达当前目标点？","Reached current target?"),"d ≤ goal_radius",ORANGE)
    f.line([(330,185),(405,185)]);f.line([(700,185),(775,185)])
    f.box(775,330,290,115,t("尚未到达：选择动作","Choose next action"),t("对准 → 前进\n偏右 / 偏左 → 相应转向","Aligned → forward\nRight / left error → turn"),PURPLE)
    f.line([(920,240),(920,330)],PURPLE)
    f.box(405,330,295,115,"Env.step(action)",t("保存动作与执行后的 RGB\n更新当前位置","Save action and resulting RGB\nUpdate current position"),GREEN)
    f.line([(775,387),(700,387)],PURPLE);f.line([(550,330),(550,240)],GREEN)
    f.box(35,330,295,115,t("已到达中间点","Intermediate arrival"),t("切换到下一个路径点\n继续计算动作","Select next waypoint\nContinue action selection"),ORANGE)
    f.line([(920,240),(920,277),(180,277),(180,330)],ORANGE)
    f.line([(180,330),(180,295),(405,295),(405,220)],ORANGE)
    f.box(370,535,695,85,t("已到达最终目标 → 执行 STOP → 保存最终帧","Final target reached → execute STOP → save final frame"),"",GREEN)
    f.line([(1065,185),(1080,185),(1080,578),(1065,578)],GREEN)
    f.text(35,490,295,110,t("转向容差随距离与上一动作调整\n减少前进 / 转向反复切换","Turn tolerance depends on distance\nand previous action to reduce\nforward / turn oscillation"),18,color=MUTED)
    f.save()

    f=Figure("frame-action-alignment",locale,t("存储对齐与下一动作监督","Stored alignment and next-action supervision"),t("示例：前进、右转、停止；最后一帧保留 STOP 执行后的观测","Example: forward, right, stop; the last frame is the observation after STOP"),560)
    xs=[45,320,595,870]
    for i,x in enumerate(xs):
        f.box(x,135,185,95,f"o{i}",f"{i+1:03d}.jpg",BLUE)
        f.box(x,315,185,90,["INIT = −1","FORWARD = 1","RIGHT = 3","STOP = 0"][i],"",ORANGE)
        f.line([(x+92,230),(x+92,250)],MUTED,False,True)
        f.line([(x+92,294),(x+92,315)],MUTED,False,True)
    for i,label in enumerate(["a₁: FORWARD","a₂: RIGHT","a₃: STOP"]):
        f.line([(xs[i]+185,181),(xs[i+1],181)],BLUE)
        f.text(xs[i]+162,95,136,34,label,15,color=BLUE)
    f.text(40,249,1020,46,t("同一列：actions[i] 记录产生 images[i] 的动作","Same column: actions[i] records the action that produced images[i]"),21)
    f.box(45,449,1010,75,t("训练配对：o₀ → a₁　　o₁ → a₂　　o₂ → a₃　｜　4 帧 = 3 个动作 + 1","Training pairs: o₀ → a₁     o₁ → a₂     o₂ → a₃   |   4 frames = 3 actions + 1"),"",GREEN)
    f.save()

    f=Figure("success-cases",locale,t("到达、停止与离开后返回","Arrival, stopping, and leave-and-return"),t("r 为当前任务的成功半径；S = Success，OS = Oracle Success","r is the task success radius; S = Success, OS = Oracle Success"),650)
    for x in [35,395,755]: f.rect(x,130,310,325,fill="#F7FAFC",stroke="#DAE4EC")
    f.text(45,145,290,60,t("到达并停止","Arrive and stop"),23,True)
    f.text(405,145,290,60,t("经过后在外部停止","Pass through, stop outside"),21,True)
    f.text(765,145,290,60,t("离开后返回","Leave and return"),23,True)
    for x in [190,550,910]:
        f.rect(x-48,245,96,96,PALE[GREEN],GREEN,ellipse=True)
        f.rect(x-5,288,10,10,GREEN,GREEN,ellipse=True)
        f.text(x-45,345,90,30,"goal / r",17,color=GREEN)
    f.line([(75,390),(135,310),(185,298)],BLUE)
    f.text(90,400,200,40,"S = 1   OS = 1",22,True,color=GREEN)
    f.line([(425,355),(550,295),(660,240)],BLUE)
    f.text(445,400,210,40,"S = 0   OS = 1",22,True,color=ORANGE)
    f.line([(910,295),(805,230),(1010,220),(1010,380),(920,305)],BLUE)
    f.text(795,400,230,40,"S = 1   OS = 1",22,True,color=GREEN)
    f.text(755,468,310,50,t("先距起点 > 2r\n再距目标 < r 并 STOP","First: distance from start > 2r\nThen: distance to goal < r + STOP"),17,color=MUTED)
    f.text(35,468,670,50,t("普通任务：在目标区域内执行 STOP 才获得 Success","Ordinary case: STOP inside the goal region gives Success"),20,color=MUTED)
    f.box(35,550,1030,70,t("离开后返回的触发条件：起点到首个目标的距离 < r","Leave-and-return applies when the start-to-first-goal distance is < r"),"",ORANGE)
    f.save()

    f=Figure("distributed-results",locale,t("Episode 分片、恢复与汇总","Episode sharding, resume and aggregation"),t("先做全局选择，再按 rank 分片；每个 worker 保存自己的 Episode 记录","Select globally, then shard by rank; each worker owns its episode log"),700)
    f.box(220,125,660,95,t("按 episode_key 排序 → offset → limit","Sort by episode_key → offset → limit"),"selected = [0, 1, 2, 3, 4, 5, 6]")
    for x,rank,eps in [(35,0,"0, 3, 6"),(395,1,"1, 4"),(755,2,"2, 5")]:
        f.box(x,305,310,130,f"rank {rank}",f"Episode {eps}\nepisodes.jsonl + done.json",BLUE)
        f.line([(550,220),(550,263),(x+155,263),(x+155,305)])
        f.line([(x+155,435),(x+155,485),(550,485),(550,535)],GREEN)
    f.box(220,535,660,80,t("等待全部 worker → aggregate → summary.json","Wait for all workers → aggregate → summary.json"),"",GREEN)
    f.text(35,635,1030,42,t("resume：读取各 rank 已有 key，继续剩余 Episode","Resume: read existing keys in each rank log, then run the remaining episodes"),18,color=MUTED)
    f.save()


if __name__ == "__main__":
    for locale in ("zh-CN","en-US"): generate(locale)
    print(f"Wrote 8 bilingual SVG/draw.io pairs to {OUT}")
