import { AppBar, Button, Toolbar } from "@material-ui/core";
import { NavLink } from "react-router-dom";

export default function Header() {
    return (
        <div>
            <h1>КиберЛес</h1>
            <AppBar position="static" color="primary">
                <Toolbar>
                    {/* <Button component={NavLink} activeClassName="active" color="inherit" to="/dashboard"><img src="logo_psyscanner.svg" width="200" alt="PSY Scanner" /></Button> */}
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/">Главная</Button>
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/regions">Регионы</Button>
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/fragments">Спутниковые снимки</Button>
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/fragments">ВС</Button>
                    <div style={{flexGrow: 1}} />
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/config">Настройки</Button>
                    <Button component={NavLink} activeClassName="active" color="inherit" to="/logout">Выход</Button>
                </Toolbar>
            </AppBar>
        </div>
    )
}