import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import "./indexaipage.css";
import "./index.css";
import "./index1.scss"
import { BrowserRouter, Switch } from "react-router-dom";
import store from "./redux/store";
import { Provider } from 'react-redux';



ReactDOM.render(<Provider store={store}>
    <BrowserRouter>
      <App />
    
    </BrowserRouter>
    </Provider>,
  document.getElementById("root")
);
