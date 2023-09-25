interface GridPos {
    x: number;
    y: number;
    h: number;
    w: number;
    /**
     * 默认为init
     */
    i: string;
}

interface Step {
    /**
     * 步骤的标题
     */
    title: string;
    /**
     * 步骤的描述
     */
    description: string;
    /**
     * 步骤的图标 默认为空字符串
     */
    icon: string;
    /**
     * 步骤的链接 默认为空字符串
     */
    href: string;
}

interface WelcomeCard {
    /**
     * 欢迎卡片的顶部主标题，按照卡片内容总结 ~必须~
     */
    title: string;
    /**
     * 表格的位置信息
     */
    gridPos: GridPos;
    /**
     * 依赖，默认为空数组
     */
    dependParams: any[];
    /**
     * style，默认为null
     */
    style: any;
    /**
     * 输出，默认为空数组
     */
    outputs: any[];
    /**
     * 刷新间隔，默认为null
     */
    refreshInterval: number | null;
    /**
     * 可见性表达式，默认为null
     */
    visibleExp: any;
    /**
     * 组件的唯一值，~比如为随机的UUID~
     */
    uniqueKey: string;
    /**
     * flex 默认为 4
     */
    flex: number;
    /**
     * 操作步骤列表
     */
    steps: Step[];
}
