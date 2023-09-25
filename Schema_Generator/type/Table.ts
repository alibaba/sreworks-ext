interface TableColumn {
    /**
     * 表格的表头内容 ~要求中文~
     */
    dataIndex: string;
    /**
     * 列的筛选器
     */
    filters?: TableColumnFilter[];
    /**
     * 列的标签
     */
    label: string;
    /**
     * 渲染函数 ~例子为"<a href='$(row.label)'>$(row.value)</a>"~
     */
    render?: string;
    /**
     * 默认排序顺序
     */
    defaultSortOrder?: "ascend" | "descend";
}

export interface TableColumnFilter {
    /**
     * 筛选器文本
     */
    text: string;
    /**
     * 筛选器的值
     */
    value: string;
    /**
     * 可能有的子筛选器
     */
    children?: TableColumnFilter[];

}

export interface TableConfig {
    /**
     * 表格的顶部主标题，按照表格内容总结 ~必须~
     */
    title: string;
    /**
     * 表格的位置信息
     */
    gridPos: {
        x: number;
        y: number;
        h: number;
        w: number;
        i: "init";
    };
    /**
     * 依赖，默认为空数组
     */
    dependParams: any[];
    /**
     * style，默认为null
     */
    style: any | null;
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
    visibleExp: any | null;
    /**
     * 组件的唯一值，~为随机的UUID~
     */
    uniqueKey: string;
    /**
     * 表格的数据源
     */
    api: {
        url: string;
        paging: boolean;
    };
    /**
     * 表格的列
     */
    columns: TableColumn[];
    /**
     * 表格的大小，默认为small
     */
    size: "small" | "medium" | "large";
}
