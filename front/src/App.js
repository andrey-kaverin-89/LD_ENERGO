// import logo from './logo.svg';
import { HashRouter as Router, Switch, Route, Redirect, Link } from "react-router-dom";
import { useHistory, withRouter } from "react-router";
import './App.css';

// import React from 'react';
// import { HashRouter as Router, Switch, Route, Redirect } from "react-router-dom";
// import { useHistory, withRouter } from "react-router";
// import { SnackbarProvider } from 'notistack';
// import moment from 'moment';
// import localization from 'moment/locale/ru';

import Front from './pages/Front'; 
import Regions from './pages/Regions'; 
import Fragments from './pages/Fragments'; 
import Header from "./components/Header";


function App() {
  return (
    <div className="App">
          {/* A <Switch> looks through its children <Route>s and
              renders the first one that matches the current URL. */}
          {/* <Header></Header> */}

          {/* <Route path="/organization"><ClientLoginPage onLogin={() => history.push('/dashboard')} /></Route> */}
      {/* <Switch>
          <Route path="/fragments" element={<Fragments />} />
          <Route path="/test" element={<Fragments />} />
          <Route path="/" element={<Fragments />} />
      </Switch> */}
      {/* <header className="App-header"> */}
        {/* <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>*/}
      {/* </header> */}

      <Router>
      <div>
        {/* <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/regions">About</Link>
          </li>
          <li>
            <Link to="/fragments">Dashboard</Link>
          </li>
        </ul>

        <hr /> */}

        {/*
          A <Switch> looks through all its children <Route>
          elements and renders the first one whose path
          matches the current URL. Use a <Switch> any time
          you have multiple routes, but you want only one
          of them to render at a time
        */}
        <Switch>
          <Route exact path="/"><Front /></Route>
          <Route path="/regions"><Regions /></Route>
          <Route path="/fragments"><Fragments /></Route>
        </Switch>
      </div>
    </Router>
    </div>
  );
}

// const AppWithRouter = withRouter(App)

// function AppWrapper() {
//   return (
//     <SnackbarProvider maxSnack={3}>
//       <Router>
//         <AppWithRouter />
//       </Router>
//     </SnackbarProvider>
//   )
// }

export default App;
