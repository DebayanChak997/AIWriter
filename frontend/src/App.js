import React from "react";
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import "../node_modules/bootstrap/dist/js/bootstrap.bundle";
import Home from "./Home";
import About from "./About";
import Contact from "./ContactPage";
// import Services from "./Service";
import Footer from "./Footer";
import Login from "./Login";
import Register from "./Register";
import Ai from "./Ai";
import Payment from "./Payment";
import Reset from "./Reset";
import ResetConfirm from "./ResetConfirm";
import Service from "./ServicePage";
// import Aipage from "./Aipage";

import { useSelector } from "react-redux";
import { Switch, Route, Redirect } from "react-router-dom";

const App = () => {
  const { userDetails } = useSelector((state) => state.userLogin);
  return (
    <>
      {/* <Navbar /> */}
      <Switch>
        <Route exact path="/" component={Home} />
        <Route exact path="/services" component={Service} />
        <Route exact path="/about" component={About} />
        <Route exact path="/contact" component={Contact} />
        <Route exact path="/login" component={Login}>
        {userDetails && <Redirect exact from="/login" to="/ai" />}</Route>
        <Route exact path="/register" component={Register} />
        <Route exact path="/ai" component={Ai}>
        {!userDetails && <Redirect exact from="/ai" to="/login" />}</Route>
        <Route exact path="/payment" component={Payment} />
        <Route exact path="/reset" component={Reset} />
        <Route exact path="/resetconfirm" component={ResetConfirm} />
        

        <Redirect to="/" />
      </Switch>
     
      <Footer/>
     
    </>
  );
};

export default App;
