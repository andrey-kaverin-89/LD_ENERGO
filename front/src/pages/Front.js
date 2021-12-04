import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { useHistory } from "react-router";
import Page from "../components/Page";


let options = {
    chart: {
        map: 'custom/world',
        borderWidth: 1
    },

    colors: ['rgba(19,64,117,0.05)', 'rgba(19,64,117,0.2)', 'rgba(19,64,117,0.4)',
        'rgba(19,64,117,0.5)', 'rgba(19,64,117,0.6)', 'rgba(19,64,117,0.8)', 'rgba(19,64,117,1)'],

    title: {
        text: 'Population density by country (/km²)'
    },

    mapNavigation: {
        enabled: true
    },

    legend: {
        title: {
            text: 'Individuals per km²',
            style: {
                color: ( // theme
                    Highcharts.defaultOptions &&
                    Highcharts.defaultOptions.legend &&
                    Highcharts.defaultOptions.legend.title &&
                    Highcharts.defaultOptions.legend.title.style &&
                    Highcharts.defaultOptions.legend.title.style.color
                ) || 'black'
            }
        },
        align: 'left',
        verticalAlign: 'bottom',
        floating: true,
        layout: 'vertical',
        valueDecimals: 0,
        backgroundColor: ( // theme
            Highcharts.defaultOptions &&
            Highcharts.defaultOptions.legend &&
            Highcharts.defaultOptions.legend.backgroundColor
        ) || 'rgba(255, 255, 255, 0.85)',
        symbolRadius: 0,
        symbolHeight: 14
    },

    colorAxis: {
        dataClasses: [{
            to: 3
        }, {
            from: 3,
            to: 10
        }, {
            from: 10,
            to: 30
        }, {
            from: 30,
            to: 100
        }, {
            from: 100,
            to: 300
        }, {
            from: 300,
            to: 1000
        }, {
            from: 1000
        }]
    },

    series: [{
        // data: {data},
        joinBy: ['iso-a2', 'code'],
        animation: true,
        name: 'Population density',
        states: {
            hover: {
                color: '#a4edba'
            }
        },
        tooltip: {
            valueSuffix: '/km²'
        },
        shadow: false
    }]
}



export default function Front() {
    const history = useHistory();
    return (
        <Page>
            <h1>Front</h1>

            <img src={'/russia2.svg'} onClick={() =>history.push('/regions')} width={'80%'} />
            {/* <HighchartsReact
                // options = { this.state.chartOptions }
                options = { options }
                highcharts = { Highcharts }
                constructorType = { 'mapChart' }
                allowChartUpdate = { true }
                immutable = { false }
                updateArgs = { [true, true, true] }
                containerProps = {{ className: 'chartContainer' }}
                callback = { this.chartCallback }
            /> */}
        </Page>
    )
}